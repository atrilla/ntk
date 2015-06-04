/*********************************************************************
  File    : linfunc.c
  Created : 04-Jun-2015
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

// Test unit with function
// y = 2x + 1

#include "Unit.h"
#include <stdio.h>

double lin(double x) {
  return x;
}

int main() {
  double in[2] = {1.0, 0.0};
  Unit *ut = UT_New(2, 0.2, lin);
  int i;
  ut->in[0] = in;
  ut->in[1] = in + 1;
  ut->w[0] = 1;
  ut->w[1] = 2;
  printf("Linear unit implementing f(x) = 2x + 1\n");
  printf("x\tf(x)\n");
  for (i = 0; i < 10; i++) {
    in[1] = (double)i;
    printf("%.4f\t%.4f\n", (double)i, UT_Eval(ut));
  }
  return 0;
}

