const KimiLLM = require("./claude-code-kimi-adapter");
const fs = require("fs");
const path = require("path");

async function reviewFile(filePath) {
  const kimi = new KimiLLM();
  
  try {
    const code = fs.readFileSync(filePath, "utf8");
    const fileName = path.basename(filePath);
    const ext = path.extname(filePath).substring(1) || "python";
    
    console.log(`\n📝 Reviewing ${fileName}...\n`);
    
    const review = await kimi.reviewCode(code, ext);
    console.log(review);
  } catch (error) {
    console.error("Error:", error.message);
  }
}

// Review the main Python files
reviewFile("/Users/ashnabaziz/Downloads/project-horizon-v18/realtime_feed.py");
