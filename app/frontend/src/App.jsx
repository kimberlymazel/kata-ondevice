import React, { useState } from "react";
import "./App.css";
import axios from "axios";

function App() {
  const [userInput, setUserInput] = useState("");
  const [assistantResponse, setAssistantResponse] = useState("");
  const [status, setStatus] = useState("idle"); // 'idle', 'listening', 'processing', 'talking'

  const handleRecordAndRespond = async () => {
    setStatus("listening");
    try {
      const recordResponse = await axios.post("http://127.0.0.1:8000/record");
      setUserInput(recordResponse.data.text);

      setStatus("processing");
      const response = await axios.post("http://127.0.0.1:8000/respond", {
        text: recordResponse.data.text,
      });
      setAssistantResponse(response.data.response);

      setStatus("talking");
      await axios.post("http://127.0.0.1:8000/speak", {
        text: response.data.response,
      });

      setStatus("idle");
    } catch (error) {
      console.error("Error during processing:", error);
      setStatus("idle");
    }
  };

  return (
    <div className="App">
      <div className={`glowing-circle ${status}`}>
        {status === "listening" && (
          <div className="wave-container">
            <div className="wave"></div>
            <div className="wave"></div>
            <div className="wave"></div>
          </div>
        )}
      </div>
      <div className="chat-interface">
        <p>
          <strong>You:</strong> {userInput}
        </p>
        <p>
          <strong>Assistant:</strong> {assistantResponse}
        </p>
        <button onClick={handleRecordAndRespond} disabled={status !== "idle"}>
          {status === "idle"
            ? "Start"
            : status === "listening"
            ? "Listening..."
            : status === "processing"
            ? "Processing..."
            : "Talking..."}
        </button>
      </div>
    </div>
  );
}

export default App;