/*********************************************************************
  File    : Network.h
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

/**
 * @brief Network (NT), general unit set.
 * @author Alexandre Trilla
 */

#ifndef NETWORK_H
#define NETWORK_H

#include "Unit.h"

typedef struct {
  Unit *ut; // Unit set.
  int nut; // Number of units.
} Network;

/**
 * @post Synchronous update.
 */
void NT_Sync(Network *nt);

/**
 * @post Asynchronous update (one random unit).
 */
void NT_Async(Network *nt);

#endif

