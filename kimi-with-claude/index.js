const OpenAI = require("openai");

const kimi = new OpenAI({
  apiKey: process.env.MOONSHOT_API_KEY,
  baseURL: "https://api.moonshot.ai/v1",
});

async function askKimi() {
  console.log("Asking Kimi a question...\n");

  const response = await kimi.chat.completions.create({
    model: "kimi-k2.5",
    messages: [
      { role: "user", content: "What is quantum computing in simple terms?" },
    ],
  });

  console.log("Kimi:", response.choices[0].message.content);
}

askKimi().catch(console.error);
