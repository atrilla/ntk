/*********************************************************************
  File    : Unit.c
  Created : 03-Jun-2015
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

#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include "Unit.h"

Unit* UT_New(int numin, double winit, fp f) {
  double r;
  int i;
  Unit *ut = malloc(sizeof(Unit));
  if (ut != NULL) {
    ut->nin = numin;
    ut->out = 0.0;
    ut->g = f;
    ut->in = malloc(numin * sizeof(double*));
    if (ut->in != NULL) {
      ut->w = malloc(numin * sizeof(double));
      if (ut->w != NULL) {
        srand(time(NULL));
        for (i = 0; i < numin; i++) {
          ut->w[i] = (((double)rand() / (double)RAND_MAX) - 0.5) *
            (winit/2);
        }
      } else {
        printf("New Unit->w failed!\n");
        free(ut->in);
        free(ut);
        exit(EXIT_FAILURE);
      }
    } else {
      printf("New Unit->in failed!\n");
      free(ut);
      exit(EXIT_FAILURE);
    }
  } else {
    printf("New Unit failed!\n");
    exit(EXIT_FAILURE);
  }
  return ut;
}

void UT_In(Unit *ut, const double *input, int idx) {
}

void UT_SetW(Unit *ut, double wval, int idx) {
}

double UT_GetW(const Unit *ut, int idx) {
}

double UT_Out(const Unit *ut) {
}

void UT_Up(Unit *ut, double y) {
}

void UT_Del(Unit *ut) {
}

