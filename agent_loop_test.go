package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

// TestDefaultCompactor verifies the behaviour of the default session compactor.
//
// Session layout (indices after Add):
//
//	[0] SystemMessage       – never touched
//	[1] UserMessage         – never touched
//	[2] ThinkingMessage     – long → truncated  (2 assistants follow)
//	[3] ThinkingMessage     – short → marked but content unchanged
//	[4] ToolCallMessage     – long input → truncated
//	[5] ToolResultMessage   – long output → truncated
//	[6] AssistantMessage    – assistant #1, never touched
//	[7] AssistantMessage    – assistant #2, never touched
func TestDefaultCompactor(t *testing.T) {
	const prefixLen = 200
	long := strings.Repeat("a", prefixLen+50)
	short := "brief"
	longInput := json.RawMessage(`"` + long + `"`) // valid JSON string, over limit

	s := Session{}
	s.Add(
		SystemMessage{"sys"},
		UserMessage{"user"},
		ThinkingMessage{Content: long},
		ThinkingMessage{Content: short},
		ToolCallMessage{ID: "c1", Name: "tool", Input: longInput},
		ToolResultMessage{ID: "c1", Output: long},
		AssistantMessage{"reply 1"},
		AssistantMessage{"reply 2"},
	)

	compact := defaultCompactor()
	s = compact(s)

	// [0] SystemMessage untouched.
	if _, ok := s.Messages[0].(SystemMessage); !ok {
		t.Fatal("[0] SystemMessage changed type")
	}

	// [1] UserMessage untouched.
	if _, ok := s.Messages[1].(UserMessage); !ok {
		t.Fatal("[1] UserMessage changed type")
	}

	// [2] Long ThinkingMessage truncated with ellipsis.
	tm := s.Messages[2].(ThinkingMessage)
	if !strings.HasSuffix(tm.Content, "…") {
		t.Errorf("[2] ThinkingMessage not compacted: %q", tm.Content)
	}
	if len(tm.Content) > prefixLen+len("…") {
		t.Errorf("[2] ThinkingMessage too long after compaction: %d bytes", len(tm.Content))
	}

	// [3] Short ThinkingMessage content unchanged.
	tm2 := s.Messages[3].(ThinkingMessage)
	if tm2.Content != short {
		t.Errorf("[3] short ThinkingMessage modified: %q", tm2.Content)
	}

	// [4] Long ToolCallMessage input compacted (JSON string containing ellipsis).
	tc := s.Messages[4].(ToolCallMessage)
	if !strings.Contains(string(tc.Input), "…") {
		t.Errorf("[4] ToolCallMessage.Input not compacted: %s", tc.Input)
	}

	// [5] Long ToolResultMessage output truncated with ellipsis.
	tr := s.Messages[5].(ToolResultMessage)
	if !strings.HasSuffix(tr.Output, "…") {
		t.Errorf("[5] ToolResultMessage not compacted: %q", tr.Output)
	}
	if len(tr.Output) > prefixLen+len("…") {
		t.Errorf("[5] ToolResultMessage too long after compaction: %d bytes", len(tr.Output))
	}

	// [6], [7] AssistantMessages untouched.
	for _, idx := range []int{6, 7} {
		am, ok := s.Messages[idx].(AssistantMessage)
		if !ok || am.Content == "" {
			t.Errorf("[%d] AssistantMessage changed", idx)
		}
	}

	// Second call: already-compacted messages must not change.
	snapshot := make([]Message, len(s.Messages))
	copy(snapshot, s.Messages)
	s = compact(s)
	for i, msg := range s.Messages {
		got, _ := json.Marshal(msg)
		want, _ := json.Marshal(snapshot[i])
		if string(got) != string(want) {
			t.Errorf("[%d] changed on second compaction:\n got  %s\n want %s", i, got, want)
		}
	}
}

// TestDefaultCompactorThreshold confirms that messages are not compacted when
// fewer than two assistant responses follow them.
func TestDefaultCompactorThreshold(t *testing.T) {
	long := strings.Repeat("a", 300)

	s := Session{}
	s.Add(
		ThinkingMessage{Content: long},  // only one assistant follows → must not compact
		ToolResultMessage{Output: long}, // same
		AssistantMessage{"only one"},    // assistant #1 — threshold not met
	)

	compact := defaultCompactor()
	s = compact(s)

	if tm := s.Messages[0].(ThinkingMessage); tm.Content != long {
		t.Errorf("ThinkingMessage should not be compacted below threshold, got %q", tm.Content)
	}
	if tr := s.Messages[1].(ToolResultMessage); tr.Output != long {
		t.Errorf("ToolResultMessage should not be compacted below threshold, got %q", tr.Output)
	}
}

// TestAgentLoopAddition runs a single-tool agent loop and confirms the model
// uses the add tool to produce a correct answer.
func TestAgentLoopAddition(t *testing.T) {
	skipIfNoKey(t)

	addTool := Tool{
		Definition: ToolDefinition{
			Name:        "add",
			Description: "Add two numbers and return their sum.",
			InputSchema: ToolInputSchema{
				Type: "object",
				Properties: map[string]any{
					"a": map[string]any{"type": "number", "description": "First operand"},
					"b": map[string]any{"type": "number", "description": "Second operand"},
				},
				Required: []string{"a", "b"},
			},
		},
		Handler: func(input json.RawMessage) (string, error) {
			var args struct {
				A float64 `json:"a"`
				B float64 `json:"b"`
			}
			if err := json.Unmarshal(input, &args); err != nil {
				return "", err
			}
			return fmt.Sprintf("%g", args.A+args.B), nil
		},
	}

	session := InitSession(
		"You are a helpful assistant. Use tools when they help.",
		"What is 1234 + 5678?",
	)

	session, err := AgentLoop(context.Background(), InvokeClaude(), []Tool{addTool}, session)
	if err != nil {
		t.Fatal(err)
	}

	for _, msg := range session.Messages {
		t.Logf("%T: %+v", msg, msg)
	}

	// Confirm the final assistant message mentions the correct answer.
	var finalReply string
	for _, msg := range session.Messages {
		if am, ok := msg.(AssistantMessage); ok {
			finalReply = am.Content
		}
	}
	if finalReply == "" {
		t.Fatal("no assistant reply in session")
	}
	if !strings.Contains(finalReply, "6912") && !strings.Contains(finalReply, "6,912") {
		t.Errorf("expected answer 6912 in reply, got: %s", finalReply)
	}
}

// TestAgentLoopSubagent demonstrates a subagent pattern: the assess_fact tool
// wraps its own AgentLoop call so the parent agent can delegate fact-grading to
// a specialised inner agent.
//
// Parent: generates science facts and calls assess_fact for each one.
// Subagent: receives a single fact, grades it on a five-point scale, and
// explains its reasoning.
func TestAgentLoopSubagent(t *testing.T) {
	skipIfNoKey(t)

	invokeModel := InvokeClaude()

	assessFactTool := Tool{
		Definition: ToolDefinition{
			Name:        "assess_fact",
			Description: "Assess how interesting a given fact is. Returns a grade and an explanation.",
			InputSchema: ToolInputSchema{
				Type: "object",
				Properties: map[string]any{
					"fact": map[string]any{
						"type":        "string",
						"description": "The fact to assess for interestingness.",
					},
				},
				Required: []string{"fact"},
			},
		},
		Handler: func(input json.RawMessage) (string, error) {
			var args struct {
				Fact string `json:"fact"`
			}
			if err := json.Unmarshal(input, &args); err != nil {
				return "", err
			}

			subSession := InitSession(
				`You are a critical expert at assessing how interesting facts are.
Before assigning a grade, briefly critique the fact: identify what makes it dull, obvious, or overly familiar to most people.
Then, weighing that critique, grade the fact using exactly one of these labels on the first line:
  not interesting | mildly interesting | interesting | very interesting | mind-bendingly interesting
Follow the grade with a short explanation that incorporates your critique and justifies the rating.`,
				fmt.Sprintf("Please assess this fact: %s", args.Fact),
			)

			subSession, err := AgentLoop(context.Background(), invokeModel, nil, subSession)
			if err != nil {
				return "", err
			}

			// Return the last assistant message produced by the subagent.
			var assessment string
			for _, msg := range subSession.Messages {
				if am, ok := msg.(AssistantMessage); ok {
					assessment = am.Content
				}
			}
			if assessment == "" {
				return "No assessment produced.", nil
			}
			return assessment, nil
		},
	}

	session := InitSession(
		"You are a knowledgeable assistant that generates interesting science facts. "+
			"For every fact you generate, you MUST call the assess_fact tool to evaluate it. "+
			"Keep generating and assessing facts until one is rated \"mind-bendingly interesting\". "+
			"Only stop once you have achieved that rating.",
		"Generate and assess science facts using the assess_fact tool until one is rated \"mind-bendingly interesting\".",
	)

	logMsg := func(msg Message) {
		data, _ := json.Marshal(msg)
		t.Logf("%T: %s", msg, data)
	}

	session, err := AgentLoop(context.Background(), invokeModel, []Tool{assessFactTool}, session,
		WithMaxIterations(5),
		WithLogger(logMsg),
	)
	if err != nil {
		t.Fatal(err)
	}

	// Confirm the parent agent actually invoked the subagent tool.
	var toolCallCount int
	for _, msg := range session.Messages {
		if tc, ok := msg.(ToolCallMessage); ok && tc.Name == "assess_fact" {
			toolCallCount++
		}
	}
	if toolCallCount == 0 {
		t.Fatal("expected at least one call to assess_fact, got none")
	}
	t.Logf("assess_fact called %d time(s)", toolCallCount)

	// Confirm that at least one tool result achieved the top grade.
	var topGradeAchieved bool
	for _, msg := range session.Messages {
		if tr, ok := msg.(ToolResultMessage); ok {
			if strings.Contains(strings.ToLower(tr.Output), "mind-bendingly interesting") {
				topGradeAchieved = true
				break
			}
		}
	}
	if !topGradeAchieved {
		t.Error("no assess_fact result achieved the \"mind-bendingly interesting\" grade")
	}
}
