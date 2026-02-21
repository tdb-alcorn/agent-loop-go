package main

import (
	"encoding/json"
	"reflect"
	"testing"
)

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
