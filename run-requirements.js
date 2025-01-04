const { exec } = require("child_process");
const fs = require("fs");

const REQUIREMENTS_FILE = "requirements.txt";

if (!fs.existsSync(REQUIREMENTS_FILE)) {
  console.error(`Error: ${REQUIREMENTS_FILE} not found.`);
  process.exit(1);
}

const pipCommand = `pip install -r ${REQUIREMENTS_FILE}`;

console.log(`Running: ${pipCommand}`);

exec(pipCommand, (error, stdout, stderr) => {
  if (error) {
    console.error(`Error executing pip install: ${error.message}`);
    process.exit(1);
  }
  if (stderr) {
    console.error(`stderr: ${stderr}`);
  }
  console.log(`stdout: ${stdout}`);
  console.log("Python dependencies installed successfully!");
});
