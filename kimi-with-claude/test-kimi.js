const KimiLLM = require("./claude-code-kimi-adapter");

async function test() {
  const kimi = new KimiLLM();
  
  const response = await kimi.chat("What is machine learning?");
  console.log(response.message);
}

test().catch(console.error);
