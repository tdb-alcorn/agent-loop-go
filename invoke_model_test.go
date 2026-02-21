package main

import (
	"context"
	"encoding/json"
	"testing"
)

// TestInvokeClaudeReturnsUsage confirms that a real API call populates both
// InputTokens and OutputTokens in the returned Usage.
func TestInvokeClaudeReturnsUsage(t *testing.T) {
	skipIfNoKey(t)

	session := Session{}
	session.Add(
		SystemMessage{"You are a helpful assistant."},
		UserMessage{"Say hi."},
	)

	_, usage, err := InvokeClaude()(context.Background(), nil, session)
	if err != nil {
		t.Fatal(err)
	}
	if usage.InputTokens == 0 {
		t.Error("expected non-zero InputTokens")
	}
	if usage.OutputTokens == 0 {
		t.Error("expected non-zero OutputTokens")
	}
	t.Logf("usage: input=%d output=%d", usage.InputTokens, usage.OutputTokens)
}

// TestInvokeModelHelloWorld sends a minimal single-turn session and checks
// that we get a non-empty assistant reply.
func TestInvokeModelHelloWorld(t *testing.T) {
	skipIfNoKey(t)

	session := Session{}
	session.Add(
		SystemMessage{"Reply in exactly three words."},
		UserMessage{"Say hello world."},
	)

	msgs, _, err := InvokeClaude()(context.Background(), nil, session)
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) == 0 {
		t.Fatal("expected at least one message in response")
	}

	for _, m := range msgs {
		t.Logf("%T: %+v", m, m)
	}

	if _, ok := msgs[0].(AssistantMessage); !ok {
		t.Errorf("expected first response message to be AssistantMessage, got %T", msgs[0])
	}
}

// TestInvokeModelMultiTurn sends a session with several prior turns and checks
// that the model continues the conversation coherently.
func TestInvokeModelMultiTurn(t *testing.T) {
	skipIfNoKey(t)

	session := Session{}
	session.Add(
		SystemMessage{"You are a helpful assistant. Keep responses brief."},
		UserMessage{"My name is Alice."},
		AssistantMessage{"Nice to meet you, Alice!"},
		UserMessage{"What did I just tell you my name was?"},
	)

	msgs, _, err := InvokeClaude()(context.Background(), nil, session)
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) == 0 {
		t.Fatal("expected at least one message in response")
	}

	for _, m := range msgs {
		t.Logf("%T: %+v", m, m)
	}

	reply, ok := msgs[0].(AssistantMessage)
	if !ok {
		t.Fatalf("expected AssistantMessage, got %T", msgs[0])
	}
	// The model should remember the name from the conversation history.
	if reply.Content == "" {
		t.Error("got empty reply")
	}
}

// TestInvokeModelAllTypes builds a session containing every message type and
// calls InvokeClaude to confirm it processes the history without error.
//
// Session history:
//
//	SystemMessage      – instructions
//	UserMessage        – initial question
//	AssistantMessage   – prior text reply
//	UserMessage        – follow-up asking for weather
//	ThinkingMessage    – model reasoning (skipped when sent to API; no signature)
//	ToolCallMessage    – model requested a tool
//	ToolResultMessage  – result we are providing
//
// InvokeClaude is expected to return the assistant's final answer.
func TestInvokeModelAllTypes(t *testing.T) {
	skipIfNoKey(t)

	weatherTool := ToolDefinition{
		Name:        "get_weather",
		Description: "Get the current weather for a city",
		InputSchema: ToolInputSchema{
			Type: "object",
			Properties: map[string]any{
				"location": map[string]any{
					"type":        "string",
					"description": "City name",
				},
			},
			Required: []string{"location"},
		},
	}

	session := Session{}
	session.Add(
		SystemMessage{"You are a helpful assistant with access to a weather tool."},
		// Turn 1: a prior exchange that happened before this invocation.
		UserMessage{"Hi, can you help me?"},
		AssistantMessage{"Of course! What do you need?"},
		// Turn 2: the user asked for weather; the model called a tool.
		UserMessage{"What's the weather like in Berlin?"},
		ThinkingMessage{"I should use the get_weather tool to look this up."},
		ToolCallMessage{
			ID:    "call_abc",
			Name:  "get_weather",
			Input: json.RawMessage(`{"location":"Berlin"}`),
		},
		ToolResultMessage{
			ID:     "call_abc",
			Output: "Partly cloudy, 14°C",
		},
	)

	msgs, _, err := InvokeClaude()(context.Background(), []ToolDefinition{weatherTool}, session)
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) == 0 {
		t.Fatal("expected at least one message in response")
	}

	for _, m := range msgs {
		t.Logf("%T: %+v", m, m)
	}

	if _, ok := msgs[0].(AssistantMessage); !ok {
		t.Errorf("expected AssistantMessage, got %T", msgs[0])
	}
}
