/*********************************************************************
  File    : Network.c
  Created : 07-Jun-2015
  By      : Alexandre Trilla <alex@atrilla.net>

  NTK - Neural Network Toolkit

  Copyright (C) 2015 Alexandre Trilla

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

#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include "Unit.h"
#include "Network.h"

void NT_Sync(Network *nt) {
  double *tmp = malloc(nt->nut * sizeof(double));
  int i;
  if (tmp != NULL) {
    for (i = 0; i < nt->nut; i++) {
      tmp[i] = UT_Eval(nt->ut + i);
    }
    for (i = 0; i < nt->nut; i++) {
      nt->ut[i].out = tmp[i];
    }
    free(tmp);
  } else {
    printf("Sync network update failed!\n");
    exit(EXIT_FAILURE);
  }
}

void NT_Async(Network *nt) {
  int r;
  static char firstime = 1;
  if (firstime) {
    srand(time(NULL));
    firstime = 0;
  }
  r = rand() % nt->nut;
  nt->ut[r].out = UT_Eval(nt->ut + r);
}

