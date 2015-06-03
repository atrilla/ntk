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
 * @post Create a new unit in heap.
 * @post Weights set to U[-winit, winit].
 * @post Output set to 0.
 */
Unit* UT_New(int numin, double winit, fp f);

/**
 * @pre input must point to allocated number.
 * @pre idx = [0 .. nin-1]
 * @post Set input ref.
 */
void UT_In(Unit *ut, const double *input, int idx);

/**
 * @pre idx = [0 .. nin-1]
 * @post Set weight for corresponding input idx.
 */
void UT_SetW(Unit *ut, double wval, int idx);

/**
 * @pre idx = [0 .. nin-1]
 * @post Get weight for corresponding input idx.
 */
double UT_GetW(const Unit *ut, int idx);

/**
 * @post Compute output.
 */
double UT_Out(const Unit *ut);

/**
 * @pre y must be a valid activation value.
 * @post Update output.
 */
void UT_Up(Unit *ut, double y);

/**
 * @post Delete (deallocate) unit.
 */
void UT_Del(Unit *ut);

#endif

