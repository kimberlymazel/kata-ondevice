import React, { useState, useEffect, useRef } from "react";
import "./App.css";
import axios from "axios";

function App() {
  const [userInput, setUserInput] = useState("");
  const [status, setStatus] = useState("idle"); // 'idle', 'listening', 'processing', 'talking'
  const [conversationHistory, setConversationHistory] = useState([
    { role: "assistant", content: "Apa yang bisa saya bantu?" },
  ]);
  const [countdown, setCountdown] = useState(0);
  const userId = "user-123";

  const chatHistoryRef = useRef(null);

  useEffect(() => {
    if (chatHistoryRef.current) {
      chatHistoryRef.current.scrollTop = chatHistoryRef.current.scrollHeight;
    }
  }, [conversationHistory]);

  const handleRecordAndRespond = async () => {
    const listeningDuration = 5; // Duration in seconds (same as backend)
    setCountdown(listeningDuration);
    setStatus("listening");

    // Countdown timer
    const interval = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          clearInterval(interval);
          setStatus("processing"); // processing immediately after listening ends
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    try {
      const recordResponse = await axios.post("http://127.0.0.1:8000/record");
      const userMessage = recordResponse.data.text;
      setUserInput(userMessage);

      setConversationHistory((prev) => {
        const updatedHistory = prev.slice();
        if (updatedHistory.length === 1 && updatedHistory[0].role === "assistant") {
          updatedHistory.shift(); // Remove the initial prompt
        }
        updatedHistory.push({ role: "user", content: userMessage });
        return updatedHistory;
      });

      const response = await axios.post("http://127.0.0.1:8000/conversation", {
        user_id: userId,
        message: userMessage,
      });

      const assistantMessage = response.data.messages.at(-1).content;

      // Add response to conversation history
      setConversationHistory((prev) => [
        ...prev,
        { role: "assistant", content: assistantMessage },
      ]);

      setStatus("talking");

      await axios.post("http://127.0.0.1:8000/speak", {
        text: assistantMessage,
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
        <div className="chat-history" ref={chatHistoryRef}>
          {conversationHistory.map((message, index) => (
            <p
              key={index}
              className={`message ${message.role === "user" ? "user" : "assistant"}`}
            >
              {message.content}
            </p>
          ))}
        </div>
        <button onClick={handleRecordAndRespond} disabled={status !== "idle"}>
          {status === "idle"
            ? "Mulai"
            : status === "listening"
            ? `Mendengarkan... (${countdown})`
            : status === "processing"
            ? "Memproses..."
            : "Berbicara..."}
        </button>
      </div>
    </div>
  );
}

export default App;