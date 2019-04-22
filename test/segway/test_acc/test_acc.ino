#include <Wire.h>

#define accel_module (0x1D)
int X,Y,Z;
char output[512];

// return centi g
int accNorm(byte d) {
    int sign = 1;
    int aux;
    int dat = (int)d;
    if (dat > 127) {
        dat = dat | 0b1111111100000000;
    }
    return dat*100/63;
}

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
    byte val;
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
                val = Wire.read();
                X = accNorm(val);
            }
            Wire.beginTransmission(accel_module);
            Wire.write(0x07); // Y
            Wire.endTransmission();
            Wire.requestFrom(accel_module, 1);
            while (Wire.available()) {
                val = Wire.read();
                Y = accNorm(val);
            }
            Wire.beginTransmission(accel_module);
            Wire.write(0x08); // Z
            Wire.endTransmission();
            Wire.requestFrom(accel_module, 1);
            while (Wire.available()) {
                val = Wire.read();
                Z = accNorm(val);
            }
            if (count > 1000) {
                sprintf(output, "X: %d, Y: %d, Z: %d \n\0", X, Y, Z);
                Serial.print(output);
                count = 0;
            }
        }
    }
}
