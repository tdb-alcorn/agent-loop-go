package agentloop

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestToolDefinitionRoundTrip(t *testing.T) {
	input := ToolDefinition{
		Name:        "get_weather",
		Description: "Get the current weather for a city.",
		InputSchema: ToolInputSchema{
			Type: "object",
			Properties: map[string]any{
				"location": map[string]any{
					"type":        "string",
					"description": "City name",
				},
				"unit": map[string]any{
					"type": "string",
					"enum": []any{"celsius", "fahrenheit"},
				},
			},
			Required: []string{"location"},
		},
	}

	data, err := json.MarshalIndent(input, "", "  ")
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	t.Log(string(data))

	var output ToolDefinition
	if err := json.Unmarshal(data, &output); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if output.Name != input.Name {
		t.Errorf("Name: got %q, want %q", output.Name, input.Name)
	}
	if output.Description != input.Description {
		t.Errorf("Description: got %q, want %q", output.Description, input.Description)
	}
	if output.InputSchema.Type != input.InputSchema.Type {
		t.Errorf("InputSchema.Type: got %q, want %q", output.InputSchema.Type, input.InputSchema.Type)
	}
	if !reflect.DeepEqual(output.InputSchema.Required, input.InputSchema.Required) {
		t.Errorf("InputSchema.Required: got %v, want %v", output.InputSchema.Required, input.InputSchema.Required)
	}
	if len(output.InputSchema.Properties) != len(input.InputSchema.Properties) {
		t.Errorf("InputSchema.Properties: got %d keys, want %d", len(output.InputSchema.Properties), len(input.InputSchema.Properties))
	}
}

func TestToolDefinitionMinimal(t *testing.T) {
	// A tool with no properties or required fields should marshal cleanly
	// without emitting null/empty fields.
	input := ToolDefinition{
		Name:        "ping",
		Description: "Check connectivity.",
		InputSchema: ToolInputSchema{Type: "object"},
	}

	data, err := json.Marshal(input)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal raw: %v", err)
	}

	schema, ok := raw["input_schema"].(map[string]any)
	if !ok {
		t.Fatal("missing input_schema")
	}
	if _, hasProps := schema["properties"]; hasProps {
		t.Error("properties should be omitted when empty")
	}
	if _, hasReq := schema["required"]; hasReq {
		t.Error("required should be omitted when empty")
	}
}

func TestSessionRoundTrip(t *testing.T) {
	input := Session{}
	input.Add(
		SystemMessage{"You are a helpful assistant."},
		UserMessage{"What's the weather in Tokyo?"},
		AssistantMessage{"Let me check that for you."},
		ThinkingMessage{"I should call the weather tool."},
		ToolCallMessage{ID: "call_1", Name: "get_weather", Input: json.RawMessage(`{"location":"Tokyo"}`)},
		ToolResultMessage{ID: "call_1", Output: "Sunny, 22°C"},
	)

	data, err := json.MarshalIndent(input, "", "  ")
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	t.Log(string(data))

	var output Session
	if err := json.Unmarshal(data, &output); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if len(output.Messages) != len(input.Messages) {
		t.Fatalf("got %d messages, want %d", len(output.Messages), len(input.Messages))
	}
	for i, got := range output.Messages {
		want := input.Messages[i]
		// Compare via re-marshaled JSON so whitespace in RawMessage fields
		// doesn't cause spurious failures.
		gotJSON, _ := json.Marshal(got)
		wantJSON, _ := json.Marshal(want)
		if !reflect.DeepEqual(gotJSON, wantJSON) {
			t.Errorf("message[%d]:\n got  %s\n want %s", i, gotJSON, wantJSON)
		}
	}
}
