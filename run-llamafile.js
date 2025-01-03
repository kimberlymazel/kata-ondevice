const { spawn } = require("child_process");
const os = require("os");
const path = require("path");

// Detect OS
const isWindows = os.platform() === "win32";

// Define the command based on the OS
const llamaCommand = isWindows
  ? "seallms-v3-1.5b-chat-q4_0.exe" // Windows
  : "./seallms-v3-1.5b-chat-q4_0"; // Unix-based executable

// Execute the process
const process = spawn(isWindows ? llamaCommand : "wine", isWindows ? [] : [llamaCommand], {
  cwd: path.resolve(__dirname), 
  stdio: "inherit",
  shell: true
});

// Handle process events
process.on("error", (err) => {
  console.error(`Failed to start LLaMA executable: ${err.message}`);
});

process.on("close", (code) => {
  if (code === 0) {
    console.log("LLaMA process finished successfully.");
  } else {
    console.error(`LLaMA process exited with code ${code}.`);
  }
});