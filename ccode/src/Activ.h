/*********************************************************************
  File    : Activ.h
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

/**
 * @brief Activation functions (AF).
 * @author Alexandre Trilla
 */

#ifndef ACTIV_H
#define ACTIV_H

/**
 * @post Linear.
 */
double AF_Linear(double x);

/**
 * @post Step, aka Heaviside.
 */
double AF_Step(double x);

#endif

