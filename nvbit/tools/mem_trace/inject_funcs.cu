/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdint.h>
#include <stdio.h>

#include "utils/utils.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the mem_access_t structure */
#include "common.h"

/* for nvbit_read_reg() */
#include "nvbit_reg_rw.h"

extern "C" __device__ __noinline__ void instrument_mem(int pred, int opcode_id,
                                                       uint64_t addr,
                                                       uint32_t reg_num,
                                                       uint64_t grid_launch_id,
                                                       uint64_t pchannel_dev) {
    /* if thread is predicated off, return */
    if (!pred) {
        return;
    }

    int active_mask = __ballot_sync(__activemask(), 1);  // get mask of all active threads & also synchronize a bit??
    const int laneid = get_laneid(); // get current laneid, only really relevant for the channel
    const int first_laneid = __ffs(active_mask) - 1; // get first laneid

	int32_t reg_val = nvbit_read_reg((uint64_t) reg_num);
	
    mem_access_t ma;

    /* collect memory address information from other threads */
    for (int i = 0; i < 32; i++) { // iterate over all 32 threads
        ma.addrs[i] = __shfl_sync(active_mask, addr, i); // returns addr of thread i if active as defined by mask
        // should be able to get data in the same way...
        ma.regs[i] = __shfl_sync(active_mask, reg_val, i);
    }

    int4 cta = get_ctaid();
    ma.grid_launch_id = grid_launch_id;
    ma.cta_id_x = cta.x;
    ma.cta_id_y = cta.y;
    ma.cta_id_z = cta.z;
    ma.warp_id = get_warpid();
    ma.opcode_id = opcode_id;

    /* first active lane pushes information on the channel */
    if (first_laneid == laneid) {
        ChannelDev* channel_dev = (ChannelDev*)pchannel_dev;
        channel_dev->push(&ma, sizeof(mem_access_t));
    }
}
