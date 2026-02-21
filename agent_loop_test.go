package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

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

	session, err := AgentLoop(context.Background(), NewClient(), []Tool{addTool}, session)
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
