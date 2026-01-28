const OpenAI = require("openai");

class KimiLLM {
  constructor(options = {}) {
    if (!process.env.MOONSHOT_API_KEY) {
      throw new Error(
        "MOONSHOT_API_KEY environment variable not set. Set it with: export MOONSHOT_API_KEY=your-key"
      );
    }

    this.client = new OpenAI({
      apiKey: process.env.MOONSHOT_API_KEY,
      baseURL: "https://api.moonshot.ai/v1",
    });

    this.model = options.model || "kimi-k2.5";
    this.temperature = options.temperature || 1;
    this.maxTokens = options.maxTokens || 4096;
    this.thinkingEnabled = options.thinking || false;
    this.thinkingBudget = options.thinkingBudget || 5000;
    this.conversationHistory = [];
  }

  async complete(prompt) {
    try {
      const response = await this.client.chat.completions.create({
        model: this.model,
        messages: [{ role: "user", content: prompt }],
        temperature: this.temperature,
        max_tokens: this.maxTokens,
        ...(this.thinkingEnabled && {
          thinking: {
            type: "enabled",
            budget_tokens: this.thinkingBudget,
          },
        }),
      });

      return response.choices[0].message.content;
    } catch (error) {
      throw new Error(`Kimi API error: ${error.message}`);
    }
  }

  async chat(userMessage) {
    try {
      this.conversationHistory.push({
        role: "user",
        content: userMessage,
      });

      const response = await this.client.chat.completions.create({
        model: this.model,
        messages: this.conversationHistory,
        temperature: this.temperature,
        max_tokens: this.maxTokens,
        ...(this.thinkingEnabled && {
          thinking: {
            type: "enabled",
            budget_tokens: this.thinkingBudget,
          },
        }),
      });

      const assistantMessage = response.choices[0].message.content;

      this.conversationHistory.push({
        role: "assistant",
        content: assistantMessage,
      });

      return {
        message: assistantMessage,
        thinking: response.choices[0].message.thinking || null,
        history: this.conversationHistory,
      };
    } catch (error) {
      throw new Error(`Kimi API error: ${error.message}`);
    }
  }

  async reviewCode(code, language = "javascript") {
    const prompt = `Review the following ${language} code and provide feedback on:
1. Code quality and best practices
2. Potential bugs or issues
3. Performance considerations
4. Suggestions for improvement

\`\`\`${language}
${code}
\`\`\``;

    return this.complete(prompt);
  }

  async generateCode(description, language = "javascript") {
    const prompt = `Generate ${language} code for the following requirement:

${description}

Provide clean, well-commented code.`;

    return this.complete(prompt);
  }

  async explainCode(code, language = "javascript") {
    const prompt = `Explain the following ${language} code in detail:

\`\`\`${language}
${code}
\`\`\`

Explain what it does, how it works, and any important concepts.`;

    return this.complete(prompt);
  }

  clearHistory() {
    this.conversationHistory = [];
  }

  enableThinking(budget = 5000) {
    this.thinkingEnabled = true;
    this.thinkingBudget = budget;
  }

  disableThinking() {
    this.thinkingEnabled = false;
  }

  getSettings() {
    return {
      model: this.model,
      temperature: this.temperature,
      maxTokens: this.maxTokens,
      thinkingEnabled: this.thinkingEnabled,
      thinkingBudget: this.thinkingBudget,
      conversationLength: this.conversationHistory.length,
    };
  }
}

module.exports = KimiLLM;
