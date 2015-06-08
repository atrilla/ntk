/*********************************************************************
  File    : xor.c
  Created : 07-Jun-2015
  By      : Alexandre Trilla <alex@atrilla.net>

  NTK - Neural Network Toolkit

  Copyright (C) 2015 A.I. Maker

  --------------------------------------------------------------------

  This file is part of NTK.

  NTK is free software: you can redistribute it and/or modify it under
  the terms of the MIT/X11 License as published by the Massachusetts
  Institute of Technology. See the MIT/X11 License for more details.

  You should have received a copy of the MIT/X11 License along with
  this source code distribution of NTK (see the COPYING file in the
  root directory).
  If not, see <http://www.opensource.org/licenses/mit-license>.

*********************************************************************/

#include "Unit.h"
#include "Activ.h"
#include "Network.h"
#include <stdio.h>

// Test network
// XOR, step activations, biased units

int main() {
  double in[2];
  double *out;
  Unit ut[6];
  Network nt;
  int i;
  double bias = 1;
  // 0 
  UT_New(ut, 1, 0.5, AF_Linear);
  ut->in[0] = in;
  ut->w[0] = 1;
  // 1
  UT_New(&ut[1], 1, 0.5, AF_Linear);
  ut[1].in[0] = in + 1;
  ut[1].w[0] = 1;
  // 2
  UT_New(&ut[2], 2, 0.5, AF_Step);
  ut[2].in[0] = &ut[0].out;
  ut[2].in[1] = &bias;
  ut[2].w[0] = 1;
  ut[2].w[1] = -0.5;
  // 3
  UT_New(&ut[3], 3, 0.5, AF_Step);
  ut[3].in[0] = &ut[0].out;
  ut[3].in[1] = &ut[1].out;
  ut[3].in[2] = &bias;
  ut[3].w[0] = 1;
  ut[3].w[1] = 1;
  ut[3].w[2] = -1.5;
  // 4
  UT_New(&ut[4], 2, 0.5, AF_Step);
  ut[4].in[0] = &ut[1].out;
  ut[4].in[1] = &bias;
  ut[4].w[0] = 1;
  ut[4].w[1] = -0.5;
  // 5
  UT_New(&ut[5], 3, 0.5, AF_Linear);
  ut[5].in[0] = &ut[2].out;
  ut[5].in[1] = &ut[3].out;
  ut[5].in[2] = &ut[4].out;
  ut[5].w[0] = 1;
  ut[5].w[1] = -2;
  ut[5].w[2] = 1;
  //
  out = &ut[5].out;
  // create network
  nt.ut = ut;
  nt.nut = 6;
  // display stuff
  printf("Sync\n");
  in[0] = 0;
  in[1] = 0;
  printf("t\tu0\tu1\tu2\tu3\tu4\tu5\n");
  for (i = 0; i < 10; i++) {
    printf("%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n", i,
      ut[0].out, ut[1].out, ut[2].out, ut[3].out, ut[4].out,
      ut[5].out);
      NT_Sync(&nt);
  }
  in[0] = 0;
  in[1] = 1;
  printf("\nt\tu0\tu1\tu2\tu3\tu4\tu5\n");
  for (i = 0; i < 10; i++) {
    printf("%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n", i,
      ut[0].out, ut[1].out, ut[2].out, ut[3].out, ut[4].out,
      ut[5].out);
      NT_Sync(&nt);
  }
  in[0] = 1;
  in[1] = 0;
  printf("\nt\tu0\tu1\tu2\tu3\tu4\tu5\n");
  for (i = 0; i < 10; i++) {
    printf("%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n", i,
      ut[0].out, ut[1].out, ut[2].out, ut[3].out, ut[4].out,
      ut[5].out);
      NT_Sync(&nt);
  }
  in[0] = 1;
  in[1] = 1;
  printf("\nt\tu0\tu1\tu2\tu3\tu4\tu5\n");
  for (i = 0; i < 10; i++) {
    printf("%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n", i,
      ut[0].out, ut[1].out, ut[2].out, ut[3].out, ut[4].out,
      ut[5].out);
      NT_Sync(&nt);
  }
  printf("Async");
  in[0] = 0;
  in[1] = 1;
  printf("\nt\tu0\tu1\tu2\tu3\tu4\tu5\n");
  for (i = 0; i < 50; i++) {
    printf("%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n", i,
      ut[0].out, ut[1].out, ut[2].out, ut[3].out, ut[4].out,
      ut[5].out);
      NT_Async(&nt);
  }
  return 0;
}

