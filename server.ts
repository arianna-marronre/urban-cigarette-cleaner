import express from "express";
import { createServer } from "http";
import { WebSocketServer, WebSocket } from "ws";
import { createServer as createViteServer } from "vite";
import path from "path";
import { fileURLToPath } from "url";
import { EV3, PORT } from "./ev3-nodejs-bt-driver.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  const server = createServer(app);
  const wss = new WebSocketServer({ server });
  const PORT_NUMBER = 3000;

  // WebSocket handling
  wss.on("connection", (ws: WebSocket) => {
    console.log("Client connected to EV3 Bridge");

    // Initialize the EV3 "real" library wrapper
    // We pass a callback that sends the generated binary packets back to the browser
    const robot = new EV3("REMOTE", (data) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(data);
      }
    });

    ws.on("message", (message) => {
      try {
        const msg = JSON.parse(message.toString());
        
        if (msg.type === "CONNECT") {
          robot.connect(() => {
            console.log("Robot logical connection established");
          });
        }

        if (msg.type === "DETECTION_UPDATE") {
          const { target, position, width } = msg.payload;
          
          if (target === "cigarette" || target === "toothbrush") {
            // "Chiamate reali della libreria": Move forward on B and C
            robot.setMotorSpeed(PORT.B | PORT.C, 60);
            robot.motorStart(PORT.B | PORT.C);
          } else if (target === "tracking") {
            // Tracking logic translated to API calls
            if (position === "LEFT") {
              robot.setMotorSpeed(PORT.B, -30);
              robot.setMotorSpeed(PORT.C, 30);
              robot.motorStart(PORT.B | PORT.C);
            } else if (position === "RIGHT") {
              robot.setMotorSpeed(PORT.B, 30);
              robot.setMotorSpeed(PORT.C, -30);
              robot.motorStart(PORT.B | PORT.C);
            } else {
              robot.setMotorSpeed(PORT.B | PORT.C, 40);
              robot.motorStart(PORT.B | PORT.C);
            }
          } else {
            // Stop
            robot.motorStop(PORT.B | PORT.C, true);
          }
        }
      } catch (e) {
        console.error("Packet processing error:", e);
      }
    });

    ws.on("close", () => {
      robot.disconnect();
      console.log("Client disconnected, robot stopped");
    });
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  server.listen(PORT_NUMBER, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT_NUMBER}`);
  });
}

startServer().catch(console.error);
