package main

import (
	"context"
	"fmt"
	"os"
	"testing"

	anthropic "github.com/anthropics/anthropic-sdk-go"
)

// TestMain loads .env before running any tests.
func TestMain(m *testing.M) {
	LoadDotEnv(".env")
	os.Exit(m.Run())
}

// skipIfNoKey skips the test when ANTHROPIC_API_KEY is not set.
func skipIfNoKey(t *testing.T) {
	t.Helper()
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}
}

// TestHelloWorld shows a basic completion.
func TestHelloWorld(t *testing.T) {
	skipIfNoKey(t)

	client := NewClaude()
	ctx := context.Background()

	msg, err := client.Complete(ctx, `Say exactly: "Hello, World!"`)
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println("=== Hello World ===")
	fmt.Println(TextContent(msg))
	fmt.Println("stop_reason:", msg.StopReason)
}

// TestThinking shows extended thinking for a reasoning-heavy problem.
func TestThinking(t *testing.T) {
	skipIfNoKey(t)

	client := NewClaude()
	ctx := context.Background()

	// Thinking requires max_tokens > budget_tokens. Budget must be >= 1024.
	msg, err := client.Complete(ctx,
		"How many r's are in the word 'strawberry'? Think carefully.",
		WithModel(anthropic.ModelClaudeSonnet4_6),
		WithThinking(4000),
		WithMaxTokens(6000),
	)
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println("=== Thinking ===")
	for _, block := range msg.Content {
		switch block.Type {
		case "thinking":
			fmt.Printf("<thinking>\n%s\n</thinking>\n\n", block.AsThinking().Thinking)
		case "text":
			fmt.Println(block.AsText().Text)
		}
	}
	fmt.Println("stop_reason:", msg.StopReason)
}

// TestWithTools shows how to define a tool and handle a tool_use response.
func TestWithTools(t *testing.T) {
	skipIfNoKey(t)

	client := NewClaude()
	ctx := context.Background()

	weatherTool := ToolDefinition{
		Name:        "get_weather",
		Description: "Get the current weather for a location",
		InputSchema: ToolInputSchema{
			Type: "object",
			Properties: map[string]any{
				"location": map[string]any{
					"type":        "string",
					"description": "City and state, e.g. San Francisco, CA",
				},
				"unit": map[string]any{
					"type":        "string",
					"enum":        []string{"celsius", "fahrenheit"},
					"description": "Temperature unit",
				},
			},
			Required: []string{"location"},
		},
	}

	msg, err := client.Complete(ctx,
		"What's the weather like in Tokyo?",
		WithTools(weatherTool),
		WithMaxTokens(1024),
	)
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println("=== With Tools ===")
	fmt.Println("stop_reason:", msg.StopReason)

	for _, tu := range ToolUseBlocks(msg) {
		fmt.Printf("tool: %s\ninput: %s\n", tu.Name, string(tu.Input))
	}
}
