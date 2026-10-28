// pressure_ttl.ino — SparkFun NAU7802 (Qwiic Scale) + TTL input, for pressure_logger.py
//
// The red board is a NAU7802 on I2C, not an HX711. Bit-banging pins 20/21 makes
// every sample 0 and floods the port at ~320 lines/s (what the last recordings did).
//
// One line per conversion, same format the logger already parses:
//   Reading: <raw> ttl=<0|1> pulses=<n> ms=<t>
//     raw     signed 24-bit NAU7802 count (channel 1, the screw terminal)
//     ttl     TTL input level at that moment
//     pulses  rising edges on the TTL input since reset (= camera frames)
//     ms      Arduino time of the sample (ms since reset)
//
// Library: Arduino IDE Library Manager ->
//   "SparkFun Qwiic Scale NAU7802 Arduino Library"
// Board: Arduino Mega or Mega 2560. Baud in the logger stays 115200.
//
// Wiring (Arduino Mega 2560)
//   NAU7802 header (I2C — these pins are fixed on the Mega):
//     VIN -> 5V     (the pin labeled VIN; 3V3 is an output, do not power it)
//     GND -> GND
//     SDA -> 20
//     SCL -> 21
//     INT     leave unconnected (this sketch polls over I2C)
//   Screw terminal: the syringe bridge stays here.
//     E+ excitation+, E- excitation-, A+ signal+, A- signal-
//     (silkscreen colors are red, black, green, white). Match the labels.
//     If a press drives the trace the wrong way, swap A+ and A-.
//   TTL  BNC T on the Inscopix camera trigger:
//     centre -> TTL_PIN, shield -> GND.  0-5 V only.
//
// Gain 128 and 80 samples/s match the logger. 320 samples/s does not fit
// this line at 115200 baud. If the trace sticks near +/-8388608, the PGA is
// saturated: lower NAU_GAIN one step and upload again.

#include <Wire.h>
#include "SparkFun_Qwiic_Scale_NAU7802_Arduino_Library.h"

const uint8_t TTL_PIN = 18;                 // Mega interrupt pins: 2, 3, 18, 19
const uint8_t NAU_GAIN = NAU7802_GAIN_128;  // 1, 2, 4, 8, 16, 32, 64, 128
const uint8_t NAU_SPS  = NAU7802_SPS_80;    // 10, 20, 40, 80. Not 320 at this baud.

NAU7802 scale;
bool scaleReady = false;

volatile unsigned long ttlPulses = 0;

void onTtlRise() { ttlPulses++; }

bool startScale() {
  // begin() powers the chip, sets the LDO to 3.3 V (that is the bridge
  // excitation on E+), and does the datasheet power-up sequence.
  if (!scale.begin()) return false;
  if (!scale.setGain(NAU_GAIN)) return false;
  if (!scale.setSampleRate(NAU_SPS)) return false;
  if (!scale.setChannel(NAU7802_CHANNEL_1)) return false;
  return scale.calibrateAFE();               // internal calibration, not a tare
}

void setup() {
  Serial.begin(115200);
  Wire.begin();                               // Mega: SDA = 20, SCL = 21
  Wire.setClock(400000);
  pinMode(TTL_PIN, INPUT_PULLUP);             // unplugged: reads 1, counts no pulses
  attachInterrupt(digitalPinToInterrupt(TTL_PIN), onTtlRise, RISING);
}

void loop() {
  if (!scaleReady) {
    scaleReady = startScale();
    if (!scaleReady) {
      Serial.println("NAU7802 not ready. Check VIN->5V, GND, SDA->20, SCL->21.");
      delay(1000);
      return;
    }
    Serial.println("NAU7802 ready");
  }

  if (!scale.available()) return;
  long raw = scale.getReading();
  unsigned long ms = millis();
  noInterrupts();
  unsigned long pulses = ttlPulses;
  interrupts();

  Serial.print("Reading: ");
  Serial.print(raw);
  Serial.print(" ttl=");
  Serial.print(digitalRead(TTL_PIN));
  Serial.print(" pulses=");
  Serial.print(pulses);
  Serial.print(" ms=");
  Serial.println(ms);
}
