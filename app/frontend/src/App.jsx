import React, { useState } from "react";
import "./App.css";
import axios from "axios";

function App() {
  const [userInput, setUserInput] = useState("");
  const [assistantResponse, setAssistantResponse] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);

  const handleRecordAndRespond = async () => {
    setIsProcessing(true); // Indicate the system is busy
    try {
      // Step 1: Record the user's speech
      const recordResponse = await axios.post("http://127.0.0.1:8000/record");
      setUserInput(recordResponse.data.text);

      // Step 2: Generate the assistant's response
      const response = await axios.post("http://127.0.0.1:8000/respond", {
        text: recordResponse.data.text,
      });
      setAssistantResponse(response.data.response);

      // Step 3: Speak the assistant's response
      await axios.post("http://127.0.0.1:8000/speak", {
        text: response.data.response,
      });

      setIsProcessing(false); // Reset the busy indicator
    } catch (error) {
      console.error("Error during processing:", error);
      setIsProcessing(false);
    }
  };

  return (
    <div className="App">
      <h1>Voice Assistant</h1>
      <button onClick={handleRecordAndRespond} disabled={isProcessing}>
        {isProcessing ? "Processing..." : "Start"}
      </button>
      <p>
        <strong>You:</strong> {userInput}
      </p>
      <p>
        <strong>Assistant:</strong> {assistantResponse}
      </p>
    </div>
  );
}

export default App;
