#include <Wire.h>

#define accel_module (0x1D)
byte x[10];
byte z[10];
char output[512];

void setup() {
    Wire.begin();
    Serial.begin(9600);
    Serial.print("Lib Setup complete\n"); 
    Wire.beginTransmission(accel_module);
    Wire.write(0x16); // MCTL
    Wire.write(0x05); // 2g, meas mode
    Wire.endTransmission();
    // 250 Hz sampling to be done
    // check this
    Serial.print("Setup complete\n"); 
}

char reg;
int count;

void loop() {
    count += 1;
    Wire.beginTransmission(accel_module);
    Wire.write(0x09); // Status
    Wire.endTransmission();
    Wire.requestFrom(accel_module, 1);
    while (Wire.available()) {
        reg = Wire.read();
        reg = reg & 0x01; // DRDY
        if (reg == 1) {
            Wire.beginTransmission(accel_module);
            Wire.write(0x06); // X
            Wire.endTransmission();
            Wire.requestFrom(accel_module, 1);
            while (Wire.available()) {
                x[0] = Wire.read();
            }
            Wire.beginTransmission(accel_module);
            Wire.write(0x08); // Z
            Wire.endTransmission();
            Wire.requestFrom(accel_module, 1);
            while (Wire.available()) {
                z[0] = Wire.read();
            }
            if (count > 1000) {
                sprintf(output, "X: %d, Z: %d \n\0", x[0], z[0]); 
                Serial.print(output);
                count = 0;
            }
        }
    }
}
