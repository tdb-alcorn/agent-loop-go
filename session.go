package main

import (
	"encoding/json"
	"fmt"
)

// Message is a sealed interface for all session turn types.
// Use a type switch to access fields of each concrete kind.
type Message interface {
	messageKind() string
}

// -- Role-based messages -------------------------------------------------

// SystemMessage carries a system-level instruction.
type SystemMessage struct{ Content string }

// UserMessage carries input from the human turn.
type UserMessage struct{ Content string }

// AssistantMessage carries a plain-text response from the model.
type AssistantMessage struct{ Content string }

// -- Content-block types ------------------------------------------------

// ThinkingMessage holds the model's internal reasoning (extended thinking).
type ThinkingMessage struct{ Content string }

// ToolCallMessage is a tool invocation requested by the model.
type ToolCallMessage struct {
	ID    string
	Name  string
	Input json.RawMessage // arbitrary JSON object
}

// ToolResultMessage is the output returned for a prior ToolCallMessage.
type ToolResultMessage struct {
	ID     string
	Output string
}

// -- Sealed-interface marker methods ------------------------------------

func (SystemMessage) messageKind() string    { return "system" }
func (UserMessage) messageKind() string      { return "user" }
func (AssistantMessage) messageKind() string { return "assistant" }
func (ThinkingMessage) messageKind() string  { return "thinking" }
func (ToolCallMessage) messageKind() string  { return "tool_call" }
func (ToolResultMessage) messageKind() string { return "tool_result" }

// -- JSON marshaling ----------------------------------------------------

func (m SystemMessage) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}{"system", m.Content})
}

func (m UserMessage) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}{"user", m.Content})
}

func (m AssistantMessage) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}{"assistant", m.Content})
}

func (m ThinkingMessage) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Type    string `json:"type"`
		Content string `json:"content"`
	}{"thinking", m.Content})
}

func (m ToolCallMessage) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Type  string          `json:"type"`
		ID    string          `json:"id"`
		Name  string          `json:"name"`
		Input json.RawMessage `json:"input"`
	}{"tool_call", m.ID, m.Name, m.Input})
}

func (m ToolResultMessage) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Type   string `json:"type"`
		ID     string `json:"id"`
		Output string `json:"output"`
	}{"tool_result", m.ID, m.Output})
}

// -- JSON unmarshaling --------------------------------------------------

// UnmarshalMessage decodes a single Message from raw JSON by inspecting
// the "role" or "type" discriminator field.
func UnmarshalMessage(data []byte) (Message, error) {
	var disc struct {
		Role string `json:"role"`
		Type string `json:"type"`
	}
	if err := json.Unmarshal(data, &disc); err != nil {
		return nil, err
	}

	type withContent struct {
		Content string `json:"content"`
	}
	type withToolCall struct {
		ID    string          `json:"id"`
		Name  string          `json:"name"`
		Input json.RawMessage `json:"input"`
	}
	type withToolResult struct {
		ID     string `json:"id"`
		Output string `json:"output"`
	}

	unmarshal := func(v any) error { return json.Unmarshal(data, v) }

	switch {
	case disc.Role == "system":
		var v withContent
		if err := unmarshal(&v); err != nil {
			return nil, err
		}
		return SystemMessage{v.Content}, nil
	case disc.Role == "user":
		var v withContent
		if err := unmarshal(&v); err != nil {
			return nil, err
		}
		return UserMessage{v.Content}, nil
	case disc.Role == "assistant":
		var v withContent
		if err := unmarshal(&v); err != nil {
			return nil, err
		}
		return AssistantMessage{v.Content}, nil
	case disc.Type == "thinking":
		var v withContent
		if err := unmarshal(&v); err != nil {
			return nil, err
		}
		return ThinkingMessage{v.Content}, nil
	case disc.Type == "tool_call":
		var v withToolCall
		if err := unmarshal(&v); err != nil {
			return nil, err
		}
		return ToolCallMessage{v.ID, v.Name, v.Input}, nil
	case disc.Type == "tool_result":
		var v withToolResult
		if err := unmarshal(&v); err != nil {
			return nil, err
		}
		return ToolResultMessage{v.ID, v.Output}, nil
	default:
		return nil, fmt.Errorf("unknown message discriminator: role=%q type=%q", disc.Role, disc.Type)
	}
}

// -- Tool definition ----------------------------------------------------

// ToolInputSchema describes the JSON schema for a tool's input parameters.
type ToolInputSchema struct {
	Type       string         `json:"type"`
	Properties map[string]any `json:"properties,omitempty"`
	Required   []string       `json:"required,omitempty"`
}

// ToolDefinition is a vendor-agnostic description of a tool the model may call.
type ToolDefinition struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema ToolInputSchema `json:"input_schema"`
}

// -- Session ------------------------------------------------------------

// Session is an ordered conversation history.
type Session struct {
	Messages []Message
}

// Add appends one or more messages to the session.
func (s *Session) Add(msgs ...Message) {
	s.Messages = append(s.Messages, msgs...)
}

func (s Session) MarshalJSON() ([]byte, error) {
	return json.Marshal(s.Messages)
}

func (s *Session) UnmarshalJSON(data []byte) error {
	var raws []json.RawMessage
	if err := json.Unmarshal(data, &raws); err != nil {
		return err
	}
	s.Messages = make([]Message, 0, len(raws))
	for _, raw := range raws {
		msg, err := UnmarshalMessage(raw)
		if err != nil {
			return err
		}
		s.Messages = append(s.Messages, msg)
	}
	return nil
}
