
/**
 * ev3-nodejs-bt - A library to control LEGO EV3 via Bluetooth Direct Commands.
 * (Custom implementation compatible with the requested API)
 */

export enum PORT {
  A = 0x01,
  B = 0x02,
  C = 0x04,
  D = 0x08,
  ALL = 0x0F
}

export class EV3 {
  private port: string;
  private connected: boolean = false;
  private onWrite: (data: Uint8Array) => void;

  constructor(port: string, onWrite: (data: Uint8Array) => void) {
    this.port = port;
    this.onWrite = onWrite;
  }

  connect(callback?: () => void) {
    this.connected = true;
    if (callback) callback();
  }

  disconnect() {
    this.connected = false;
  }

  private send(command: number[]) {
    // EV3 Direct Command Header: 
    // Message Size (2 bytes) + Message Counter (2 bytes) + Command Type (1 byte) + Header (2 bytes)
    const header = [
      command.length + 2, // size low
      0x00,               // size high
      0x01, 0x00,         // counter
      0x80,               // Direct command NO_REPLY
      0x00, 0x00          // layers/globals
    ];
    const full = new Uint8Array([...header, ...command]);
    this.onWrite(full);
  }

  // API Methods
  setMotorSpeed(ports: number, speed: number) {
    // Op: opOutputSpeed (0xA4), Layer: 0x00, Ports: ports, Speed: speed (signed byte)
    // Speed conversion for EV3: -100 to 100
    const s = speed > 100 ? 100 : (speed < -100 ? -100 : speed);
    this.send([0xA4, 0x00, ports, s & 0xFF]);
  }

  motorStart(ports: number) {
    // Op: opOutputStart (0xA6), Layer: 0x00, Ports: ports
    this.send([0xA6, 0x00, ports]);
  }

  motorStop(ports: number, brake: boolean = true) {
    // Op: opOutputStop (0xA3), Layer: 0x00, Ports: ports, Brake: 0=float, 1=brake
    this.send([0xA3, 0x00, ports, brake ? 0x01 : 0x00]);
  }

  motorStepSpeed(ports: number, speed: number, rampUp: number, step: number, rampDown: number, brake: boolean = true) {
    // Simplified step speed
    this.send([
        0xAE, 0x00, ports, speed & 0xFF, 
        rampUp & 0xFF, (rampUp >> 8) & 0xFF,
        step & 0xFF, (step >> 8) & 0xFF,
        rampDown & 0xFF, (rampDown >> 8) & 0xFF,
        brake ? 0x01 : 0x00
    ]);
  }
}
