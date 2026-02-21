package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"
	"testing"

	anthropic "github.com/anthropics/anthropic-sdk-go"
)

// TestMain loads .env before running any tests.
func TestMain(m *testing.M) {
	loadDotEnv(".env")
	os.Exit(m.Run())
}

// loadDotEnv reads KEY=VALUE pairs from path and calls os.Setenv for each.
// Lines starting with '#' and blank lines are ignored.
// Existing env vars are not overwritten.
func loadDotEnv(path string) {
	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, val, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		val = strings.TrimSpace(val)
		if os.Getenv(key) == "" {
			os.Setenv(key, val)
		}
	}
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

	client := NewClient()
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

	client := NewClient()
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

	client := NewClient()
	ctx := context.Background()

	// Build a tool definition using the SDK constructor.
	weatherTool := anthropic.ToolUnionParamOfTool(
		anthropic.ToolInputSchemaParam{
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
		"get_weather",
	)
	weatherTool.OfTool.Description = anthropic.String("Get the current weather for a location")

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
