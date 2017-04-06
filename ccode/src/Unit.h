/*********************************************************************
  File    : Unit.h
  Created : 01-Jun-2015
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

/**
 * @brief Unit (UT), general computation element.
 * @author Alexandre Trilla
 */

#ifndef UNIT_H
#define UNIT_H

typedef double (*fp)(double x); // Function pointer.

typedef struct {
  double **in; // List of input refs.
  int nin; // Number of inputs.
  double *w; // Input weights.
  double out; // Output.
  fp g; // Activation function.
} Unit;

/**
 * @pre numin >= 1
 * @pre 0 < winit < 1
 * @pre f must be a valid activation function.
 * @post Init a new unit.
 * @post Weights set to U[-winit, winit].
 * @post Output set to 0.
 */
void UT_New(Unit *ut, int numin, double winit, fp f);

/**
 * @post Delete (deallocate) unit.
 */
void UT_Del(Unit *ut);

/**
 * @post Evaluate unit.
 */
double UT_Eval(Unit *ut);

#endif

