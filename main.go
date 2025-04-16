package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/invopop/jsonschema"
	"github.com/ollama/ollama/api"
)

const MODEL_NAME = "qwen2.5-coder:32b-instruct-q8_0"
const OLLAMA_HOST = "localhost:11435"

var (
	True  = true
	False = false
)

func main() {
	ollamaHost := os.Getenv("OLLAMA_HOST")
	if len(ollamaHost) <= 0 {
		os.Setenv("OLLAMA_HOST", OLLAMA_HOST)
	}
	client, err := api.ClientFromEnvironment()
	if err != nil {
		fmt.Printf("Error: %s\n", err.Error())
	}

	scanner := bufio.NewScanner(os.Stdin)
	getUserMessage := func() (string, bool) {
		if !scanner.Scan() {
			return "", false
		}
		return scanner.Text(), true
	}

	tools := []ToolDefinition{ReadFileDefintion, ListFilesDefinition, EditFileDefinition}
	agent := NewAgent(client, getUserMessage, tools)
	err = agent.Run(context.TODO())
	if err != nil {
		fmt.Printf("Error: %s\n", err.Error())
	}
}

func NewAgent(client *api.Client, getUserMessage func() (string, bool), tools []ToolDefinition) *Agent {
	return &Agent{
		client:         client,
		getUserMessage: getUserMessage,
		tools:          tools,
	}
}

// Agent and methods
type Agent struct {
	client         *api.Client
	getUserMessage func() (string, bool)
	tools          []ToolDefinition
}

func (a *Agent) Run(ctx context.Context) error {
	conversation := []api.Message{}

	fmt.Println("Chat with Ollama (use 'ctrl-c' to quit)")

	for {
		fmt.Print("\u001b[94mYou\u001b[0m: ")
		userInput, ok := a.getUserMessage()
		if !ok {
			break
		}

		userMessage := api.Message{
			Role:    "user",
			Content: userInput,
		}
		conversation = append(conversation, userMessage)

		fmt.Print("\u001b[93mOllama\u001b[0m: ")

		response := a.runInference(ctx, conversation)
		conversation = append(conversation, response)

		for {
			if response.Role != "tool" {
				break
			}
			response = a.runInference(ctx, conversation)
			conversation = append(conversation, response)
		}

		fmt.Print("\n")
	}

	return nil
}

func (a *Agent) runInference(ctx context.Context, conversation []api.Message) api.Message {
	responseTokens := []string{}
	var responder string
	handler := func(chat api.ChatResponse) error {
		message := chat.Message
		if message.ToolCalls != nil {
			for _, call := range message.ToolCalls {
				toolResp := a.executeTool(call.Function.Name, call.Function.Arguments)
				responder = toolResp.Role
				responseTokens = append(responseTokens, toolResp.Content)
			}
		}
		if message.Content != "" {
			fmt.Print(message.Content)
			responseTokens = append(responseTokens, message.Content)
			responder = chat.Message.Role
		}

		return nil
	}

	ollamaTools := []api.Tool{}
	for _, tool := range a.tools {
		ollamaTools = append(ollamaTools, api.Tool{
			Type: "function",
			Function: api.ToolFunction{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  tool.InputSchema,
			},
		})
	}
	chatRequest := &api.ChatRequest{
		Model:    MODEL_NAME,
		Messages: conversation,
		Tools:    ollamaTools,
		Stream:   &False,
	}
	err := a.client.Chat(ctx, chatRequest, handler)
	if err != nil {
		panic(err)
	}

	response := removeToolTokens(strings.Join(responseTokens, " "))
	responseMsg := api.Message{
		Content: response,
		Role:    responder,
	}
	return responseMsg
}

func (a *Agent) executeTool(name string, arguments api.ToolCallFunctionArguments) api.Message {
	found := false
	var toolDef ToolDefinition
	for _, tool := range a.tools {
		if tool.Name == name {
			found = true
			toolDef = tool
		}
	}

	if !found {
		return api.Message{
			Role:    "tool",
			Content: "tool not found",
		}
	}

	fmt.Printf("\u001b[92mtool\u001b[0m: %s(%s)\n", name, arguments)
	response, err := toolDef.Function(arguments)
	if err != nil {
		return api.Message{
			Role:    "tool",
			Content: err.Error(),
		}
	}

	return api.Message{
		Role:    "tool",
		Content: response,
	}
}

// ToolTypes
type ToolDefinition struct {
	Name        string             `json:"name"`
	Description string             `json:"description"`
	InputSchema ToolFunctionParams `json:"input_schema"`
	Function    func(input map[string]any) (string, error)
}

type ToolFunctionParams struct {
	Type       string   `json:"type"`
	Required   []string `json:"required"`
	Properties map[string]struct {
		Type        string   `json:"type"`
		Description string   `json:"description"`
		Enum        []string `json:"enum,omitempty"`
	} `json:"properties"`
}

// ToolDefinitions
var ReadFileDefintion = ToolDefinition{
	Name:        "read_file",
	Description: "Read the contents of a given relative file path. Use this when you want to see what's inside a file. Do not use this with directory names.",
	InputSchema: ReadFileInputSchema,
	Function:    ReadFile,
}

type ReadFileInput struct {
	Path string `json:"path" jsonschema_description:"The relative path of a file in the working directory."`
}

var ReadFileInputSchema = GenerateSchema[ReadFileInput]()

func ReadFile(input map[string]any) (string, error) {
	path, ok := input["path"].(string)
	if !ok {
		panic("Attempted to call ReadFile with invalid input")
	}

	content, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}

	return string(content), nil
}

var ListFilesDefinition = ToolDefinition{
	Name:        "list_files",
	Description: "List files and directories at a given path. If no path is provided, lists files in the current directory.",
	InputSchema: ListFilesInputSchema,
	Function:    ListFiles,
}

type ListFilesInput struct {
	Path string `json:"path,omitempty" jsonschema_description:"Optional relative path to list files from. Defaults to current directory if not provided."`
}

var ListFilesInputSchema = GenerateSchema[ListFilesInput]()

func ListFiles(input map[string]any) (string, error) {
	path, ok := input["path"].(string)
	if !ok {
		path = ""
	}
	dir := "."
	if path != "" {
		dir = path
	}

	var files []string
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		relPath, err := filepath.Rel(dir, path)
		if err != nil {
			return err
		}

		if relPath != "." {
			if info.IsDir() {
				files = append(files, relPath+"/")
			} else {
				files = append(files, relPath)
			}
		}
		return nil
	})

	if err != nil {
		return "", err
	}

	result, err := json.Marshal(files)
	if err != nil {
		return "", err
	}

	return string(result), nil
}

var EditFileDefinition = ToolDefinition{
	Name: "edit_file",
	Description: `Make edits to a text file.

Replaces 'old_str' with 'new_str' in the given file. 'old_str' and 'new_str' MUST be different from each other.

If the file specified with path doesn't exist, it will be created.
`,
	InputSchema: EditFileInputSchema,
	Function:    EditFile,
}

type EditFileInput struct {
	Path   string `json:"path" jsonschema_description:"The path to the file"`
	OldStr string `json:"old_str" jsonschema_description:"Text to search for - must match exactly and must only have one match exactly"`
	NewStr string `json:"new_str" jsonschema_description:"Text to replace old_str with"`
}

var EditFileInputSchema = GenerateSchema[EditFileInput]()

func EditFile(input map[string]any) (string, error) {
	path, ok := input["path"].(string)
	if !ok {
		return "", fmt.Errorf("Did not provide a path properly")
	}
	oldStr, ok := input["old_str"].(string)
	if !ok {
		return "", fmt.Errorf("Did not provide a old_str properly")
	}
	newStr, ok := input["new_str"].(string)
	if !ok {
		return "", fmt.Errorf("Did not provide a new_str properly")
	}

	content, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) && oldStr == "" {
			return createNewFile(path, newStr)
		}
		return "", err
	}

	oldContent := string(content)
	newContent := strings.Replace(oldContent, oldStr, newStr, -1)

	if oldContent == newContent && oldStr != "" {
		return "", fmt.Errorf("old_str not found in file")
	}

	err = os.WriteFile(path, []byte(newContent), 0644)
	if err != nil {
		return "", err
	}

	return "OK", nil
}

// Utils

func validateStringArr(input []any) []string {
	ret := []string{}
	for _, item := range input {
		str, ok := item.(string)
		if !ok {
			fmt.Printf("Not a string item encountered")
			continue
		}
		ret = append(ret, str)
	}
	return ret
}

func GenerateSchema[T any]() ToolFunctionParams {
	reflector := jsonschema.Reflector{
		AllowAdditionalProperties: false,
		DoNotReference:            true,
	}
	var v T

	schema := reflector.Reflect(v)
	properties := make(map[string]struct {
		Type        string   `json:"type"`
		Description string   `json:"description"`
		Enum        []string `json:"enum,omitempty"`
	})
	for pair := schema.Properties.Oldest(); pair != nil; pair = pair.Next() {
		enum := validateStringArr(pair.Value.Enum)
		properties[pair.Key] = struct {
			Type        string   `json:"type"`
			Description string   `json:"description"`
			Enum        []string `json:"enum,omitempty"`
		}{
			Type:        pair.Value.Type,
			Description: pair.Value.Description,
			Enum:        enum,
		}
	}

	return ToolFunctionParams{
		Type:       "object",
		Required:   schema.Required,
		Properties: properties,
	}
}

func removeToolTokens(response string) string {
	// Define tool tokens to remove
	toolTokens := []string{"<|im_start|>", "<|im_end|>"}

	for _, token := range toolTokens {
		response = strings.ReplaceAll(response, token, "")
	}

	return response
}

func createNewFile(filePath, content string) (string, error) {
	dir := path.Dir(filePath)
	if dir != "." {
		err := os.MkdirAll(dir, 0755)
		if err != nil {
			return "", fmt.Errorf("failed to create directory: %w", err)
		}
	}

	err := os.WriteFile(filePath, []byte(content), 0644)
	if err != nil {
		return "", fmt.Errorf("failed to create file: %w", err)
	}

	return fmt.Sprintf("Successfully created file %s", filePath), nil
}
