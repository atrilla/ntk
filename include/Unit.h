/*********************************************************************
  File    : Unit.h
  Created : 01-Jun-2015
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

/**
 * @brief Unit (UT), general computation element.
 * @author Alexandre Trilla
 */

#ifndef UNIT_H
#define UNIT_H

typedef struct {
  double **in; // List of input refs.
  int nin; // Number of inputs.
  double *w; // Input weights.
  double out; // Output.
  double (*g)(double x); // Activation function.
} Unit;

/**
 * @post Compute the weighted sum of inputs.
 */
double UT_WSum(Unit *ut);

#endif

