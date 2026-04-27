#include "build-fast.h"
#include "build.h"
#include "bvh.h"
#include "util.h"
#include "vec_math_helper.h"

#include <cub/device/device_radix_sort.cuh>
#include <sutil/vec_math.h>


// This is a copy of build.cu. Modify it to be faster.
// Gets compiled to cuda-lbvh-fast

#define MERGING_THRESHOLD 16

struct float2x3 {
    float3 bounds[2];

    __host__ __device__ float3 &operator[](int i) { return bounds[i]; }
    __host__ __device__ const float3 &operator[](int i) const { return bounds[i]; }
};

struct cluster {
    int node_idx;
    float3 min;
    float3 max;
    int active;
};

/* \brief Interleave the first 10 bits of x every three bits,
 * ie insert two zeroes between every of the first 10 bits of x
 * \param x Quantitized position, must be between 0 and 2^10 - 1 = 1023
 */

__device__ int get_leaf_node_idx(int i, int num_triangles) { return num_triangles - 1 + i; }

__device__ int get_internal_node_idx(int i) { return i; }

/// Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
__forceinline__ __device__ unsigned int expand_bits(unsigned int v) {
    /* Comments generated with Python from https://stackoverflow.com/questions/18529057/produce-interleaving-bit-patterns-morton-keys-for-32-bit-64-bit-and-128bit */

    /*
     * Current Mask:           0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0011 1111 1111
     * Which bits to shift:    0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0011 0000 0000  hex: 0x300
     * Shifted part (<< 16):   0000 0000 0000 0000 0000 0000 0000 0000 0000 0011 0000 0000 0000 0000 0000 0000  hex: 0x3000000
     * NonShifted Part:        0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 1111 1111  hex: 0xff
     * Bitmask is now :        0000 0000 0000 0000 0000 0000 0000 0000 0000 0011 0000 0000 0000 0000 1111 1111  hex: 0x30000ff
     */
    v = (v | (v << 16)) & 0x30000ff;

    /*
     * Current Mask:           0000 0000 0000 0000 0000 0000 0000 0000 0000 0011 0000 0000 0000 0000 1111 1111
     * Which bits to shift:    0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 1111 0000  hex: 0xf0
     * Shifted part (<< 8):    0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 1111 0000 0000 0000  hex: 0xf000
     * NonShifted Part:        0000 0000 0000 0000 0000 0000 0000 0000 0000 0011 0000 0000 0000 0000 0000 1111  hex: 0x300000f
     * Bitmask is now :        0000 0000 0000 0000 0000 0000 0000 0000 0000 0011 0000 0000 1111 0000 0000 1111  hex: 0x300f00f
     */
    v = (v | (v << 8)) & 0x300f00f;

    /*
     * Current Mask:           0000 0000 0000 0000 0000 0000 0000 0000 0000 0011 0000 0000 1111 0000 0000 1111
     * Which bits to shift:    0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 1100 0000 0000 1100  hex: 0xc00c
     * Shifted part (<< 4):    0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 1100 0000 0000 1100 0000  hex: 0xc00c0
     * NonShifted Part:        0000 0000 0000 0000 0000 0000 0000 0000 0000 0011 0000 0000 0011 0000 0000 0011  hex: 0x3003003
     * Bitmask is now :        0000 0000 0000 0000 0000 0000 0000 0000 0000 0011 0000 1100 0011 0000 1100 0011  hex: 0x30c30c3
     */
    v = (v | (v << 4)) & 0x30c30c3;

    /*
     * Current Mask:           0000 0000 0000 0000 0000 0000 0000 0000 0000 0011 0000 1100 0011 0000 1100 0011
     * Which bits to shift:    0000 0000 0000 0000 0000 0000 0000 0000 0000 0010 0000 1000 0010 0000 1000 0010  hex: 0x2082082
     * Shifted part (<< 2):    0000 0000 0000 0000 0000 0000 0000 0000 0000 1000 0010 0000 1000 0010 0000 1000  hex: 0x8208208
     * NonShifted Part:        0000 0000 0000 0000 0000 0000 0000 0000 0000 0001 0000 0100 0001 0000 0100 0001  hex: 0x1041041
     * Bitmask is now :        0000 0000 0000 0000 0000 0000 0000 0000 0000 1001 0010 0100 1001 0010 0100 1001  hex: 0x9249249
     */
    v = (v | (v << 2)) & 0x9249249;

    return v;
}

/// Calculates a 30-bit Morton code for the given 3D point located
/// within the unit cube [0,1].
__forceinline__ __device__ unsigned int morton_3d(float x, float y, float z) {
    return expand_bits(x) | expand_bits(y) << 1 | expand_bits(z) << 2;
}

// For every triangle, assign a Morton code based on its center.
__global__ void assign_morton(
    const float3 *positions,
    const int *pos_indices,
    float3 scene_offset,
    float3 scene_extent,
    unsigned int *d_morton,
    unsigned int *d_ids,
    unsigned int object_count) {
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= object_count)
        return;

    // obtain center of triangle
    int idx_u = pos_indices[3 * thread_id + 0];
    int idx_v = pos_indices[3 * thread_id + 1];
    int idx_w = pos_indices[3 * thread_id + 2];
    float3 pos = (1.f / 3.f) * (positions[idx_u] + positions[idx_v] + positions[idx_w]);

    // normalize position
    float x = (pos.x - scene_offset.x) / scene_extent.x;
    float y = (pos.y - scene_offset.y) / scene_extent.y;
    float z = (pos.z - scene_offset.z) / scene_extent.z;
    // clamp to deal with numeric issues
    x = fclampf(x, 0.f, 1.f);
    y = fclampf(y, 0.f, 1.f);
    z = fclampf(z, 0.f, 1.f);

    // obtain and set morton code based on normalized position
    d_morton[thread_id] = morton_3d(x, y, z);

    d_ids[thread_id] = thread_id;
}

// todo this kernel is pretty small, can it be combined with another?
__global__ void leaf_nodes(
    unsigned int *sorted_object_ids, unsigned int num_objects, bvh_node *nodes) {
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= num_objects)
        return;

    bvh_node *internal_nodes = nodes;
    bvh_node *leaf_nodes = nodes + num_objects - 1;

    // no need to set parent to nullptr, each child will have a parent
    leaf_nodes[thread_id].object_id = sorted_object_ids[thread_id];
    // needed to recognize that this node is a leafF
    leaf_nodes[thread_id].child_l = -1;

    // Need to set for internal node parent to nullptr, to detect the root node.
    // There is one less internal node than leaf node, test for that.
    if (thread_id >= num_objects - 1)
        return;
    internal_nodes[thread_id].paren = -1;
}

__forceinline__ __device__ int delta(int l, int r, unsigned int n, unsigned int *c, unsigned int kl) {
    // this guard is for leaf nodes, not internal nodes (hence [0, n-1])
    if (r < 0 || r > n - 1)
        return -1;
    unsigned int kr = c[r];
    if (kl == kr) {
        // if keys are equal, use id as fallback
        // (+32 because they have the same morton code)
        return 32 + __clz(static_cast<unsigned int>(l) ^ static_cast<unsigned int>(r));
    }
    // clz = count leading zeros
    return __clz(kl ^ kr);
}

static __forceinline__ __device__ uint64_t fast_delta(unsigned int a, unsigned int b, unsigned int* morton_codes){
    return ((uint64_t)morton_codes[a] << 32 | a)  ^ ((uint64_t)morton_codes[b] << 32 | b);
}

// Load float3 at global level (cache in L2 and below, not L1).
__device__ float3 __ldcg(const float3 *p) {
    return make_float3(__ldcg(&(p->x)), __ldcg(&(p->y)), __ldcg(&(p->z)));
}

// Store float3 at global level (cache in L2 and below, not L1).
__device__ void __stcg(float3 *p, const float3 &q) {
    __stcg(&(p->x), q.x);
    __stcg(&(p->y), q.y);
    __stcg(&(p->z), q.z);
}

__host__ __device__ float2x3 make_bounds(float3 min, float3 max) {
    float2x3 result;
    result[0] = min;
    result[1] = max;
    return result;
}

__host__ __device__ float2x3 grow(float2x3 a, float2x3 b) {
    float2x3 result;

    result[0] = make_float3(fminf(a[0].x, b[0].x),
                            fminf(a[0].y, b[0].y),
                            fminf(a[0].z, b[0].z));

    result[1] = make_float3(fmaxf(a[1].x, b[1].x),
                            fmaxf(a[1].y, b[1].y),
                            fmaxf(a[1].z, b[1].z));

    return result;
}

__host__ __device__ float Area(float2x3 a) {
    float3 diff = fmaxf(a[1] - a[0], make_float3(0.f));
    return 2.f * (diff.x * diff.y + diff.x * diff.z + diff.y * diff.z);
}

__host__ __device__ float surface_area(float3 min, float3 max) {
    return Area(make_bounds(min, max));
}

__host__ __device__ float surface_area(const cluster &c) {
    return surface_area(c.min, c.max);
}

__device__ float3 shfl_sync_float3(unsigned int mask, float3 value, int src_lane) {
    return make_float3(
        __shfl_sync(mask, value.x, src_lane),
        __shfl_sync(mask, value.y, src_lane),
        __shfl_sync(mask, value.z, src_lane));
}

__device__ float2x3 shfl_sync_float2x3(unsigned int mask, float2x3 value, int src_lane) {
    return make_bounds(
        shfl_sync_float3(mask, value[0], src_lane),
        shfl_sync_float3(mask, value[1], src_lane));
}

__device__ uint2 shfl_sync_uint2(unsigned int mask, uint2 value, int src_lane) {
    return make_uint2(
        __shfl_sync(mask, value.x, src_lane),
        __shfl_sync(mask, value.y, src_lane));
}

static __forceinline__ __device__ uint32_t load_indices(uint32_t start, uint32_t end, uint32_t &cluster_index, build_state &state, uint32_t offset) {

    uint32_t lane_warp_index = threadIdx.x & (WARP_SIZE - 1);

    uint32_t index = lane_warp_index - offset;
    bool valid_id = index < min(end - start, MERGING_THRESHOLD);

    if (valid_id) {
        cluster_index = state.cluster_indicies[start + index];
    }

    uint32_t valid_cluster_num = __popc(__ballot_sync(FULL_MASK, valid_id && cluster_index != INVALID_IDX));

    return valid_cluster_num;
}

static __forceinline__ __device__ void store_indicies(uint32_t previous_prim, uint32_t cluster_index, build_state &state, uint32_t left_start) {

    uint32_t lane_warp_index = threadIdx.x & (WARP_SIZE - 1);

    if (lane_warp_index < previous_prim) {
        state.cluster_indicies[left_start + lane_warp_index] = cluster_index;
    }

    __threadfence();
}

static __forceinline__ __device__ uint32_t find_parent_id(unsigned int left, unsigned int right, unsigned int primCount, unsigned int *sorted_codes) {
    if (left == 0 || (right != primCount - 1 && fast_delta(right, right + 1, sorted_codes) < fast_delta(left - 1, left, sorted_codes)))
        return right;
    else
        return left - 1;
}

static inline __device__ uint32_t find_nearest_neighbor(uint32_t numPrim, float2x3 cluster_bounds) {

    uint32_t warp_id = threadIdx.x & (WARP_SIZE - 1);

    uint2 min_area_index = make_uint2(INVALID_IDX, INVALID_IDX);

    for (unsigned short r = 1; r <= SEARCH_RADIUS; r++) {
        uint32_t neighbor_index = warp_id + r;
        uint32_t area = (uint32_t)(-1);
        const int neighbor_lane = static_cast<int>(neighbor_index);
        const int src_lane = neighbor_lane < WARP_SIZE ? neighbor_lane : static_cast<int>(warp_id);

        float2x3 neighbor_bounds = shfl_sync_float2x3(FULL_MASK, cluster_bounds, src_lane);

        if (neighbor_lane < WARP_SIZE && neighbor_index < numPrim) {
            neighbor_bounds = grow(neighbor_bounds, cluster_bounds);

            area = __float_as_uint(Area(neighbor_bounds));

            if (area < min_area_index.x) {
                min_area_index = make_uint2(area, neighbor_index);
            }
        }

        uint2 neighbor_nn = shfl_sync_uint2(FULL_MASK, min_area_index, src_lane);

        if (area < neighbor_nn.x) {
            neighbor_nn = make_uint2(area, warp_id);
        }

        const int back_lane = static_cast<int>(warp_id) - r;
        const int dst_lane = back_lane >= 0 ? back_lane : static_cast<int>(warp_id);
        const uint2 back_neighbor_nn = shfl_sync_uint2(FULL_MASK, neighbor_nn, dst_lane);
        if (back_lane >= 0)
            min_area_index = back_neighbor_nn;
    }

    return min_area_index.y;
}

static inline __device__ uint32_t merge_clusters(uint32_t n_prim, uint32_t neighbor, uint32_t &cluster_index, float2x3 &cluster_bounds, build_state &state) {

    uint32_t warp_id = threadIdx.x & (WARP_SIZE - 1);

    bool active = warp_id < n_prim;

    uint32_t neighbor_warp_id = __shfl_sync(FULL_MASK, neighbor, neighbor);
    bool has_mutual_lane = active && warp_id == neighbor_warp_id;
    bool can_merge = has_mutual_lane && warp_id < neighbor;

    uint32_t merge_mask = __ballot_sync(FULL_MASK, can_merge);
    uint32_t new_nodes = __popc(merge_mask);

    uint32_t global_base_idx;
    if (warp_id == 0)
        global_base_idx = atomicAdd(state.cluster_count, new_nodes);

    global_base_idx = __shfl_sync(FULL_MASK, global_base_idx, 0);

    uint32_t relative_idx = __popc(merge_mask << (WARP_SIZE - warp_id));

    uint32_t neighbor_cluster_index = __shfl_sync(FULL_MASK, cluster_index, neighbor);
    float2x3 neighbor_bounds = shfl_sync_float2x3(FULL_MASK, cluster_bounds, neighbor);

    if (can_merge) {
        cluster_bounds = grow(cluster_bounds, neighbor_bounds);
        uint32_t new_idx = global_base_idx + relative_idx;

        bvh_node *node = &state.nodes[new_idx];
        node->child_l = cluster_index;
        node->child_r = neighbor_cluster_index;
        node->min = cluster_bounds[0];
        node->max = cluster_bounds[1];
        node->visited = 0;
        node->paren = -1;

        state.nodes[cluster_index].paren = new_idx;
        state.nodes[neighbor_cluster_index].paren = new_idx;

        cluster_index = new_idx;
    }

    uint32_t valid_mask = __ballot_sync(FULL_MASK, can_merge || !has_mutual_lane);
    int32_t shift = __fns(valid_mask, 0, warp_id + 1);
    
    cluster_index = __shfl_sync(FULL_MASK, cluster_index, shift);
    if (shift == -1)
        cluster_index = INVALID_IDX;

    cluster_bounds = shfl_sync_float2x3(FULL_MASK, cluster_bounds, shift);

    return n_prim - new_nodes;
}

static inline __device__ void ploc_merge(unsigned int lane_id, uint32_t left, uint32_t right, uint32_t split, bool final, build_state &state) {

    uint32_t left_start = __shfl_sync(FULL_MASK, left, lane_id);
    uint32_t right_end = __shfl_sync(FULL_MASK, right, lane_id) + 1;
    uint32_t left_end = __shfl_sync(FULL_MASK, split, lane_id);
    uint32_t right_start = left_end + 1;

    uint32_t lane_warp_index = threadIdx.x & (WARP_SIZE - 1);

    uint32_t cluster_index = INVALID_IDX;

    uint32_t num_left = load_indices(left_start, left_end, cluster_index, state, 0);
    uint32_t num_right = load_indices(right_start, right_end, cluster_index, state, num_left);
    uint32_t numPrim = num_left + num_right;

    float2x3 cluster_bounds;

    if (lane_warp_index < numPrim) {
        cluster_bounds = make_bounds(state.nodes[cluster_index].min, state.nodes[cluster_index].max);
    }

    uint32_t threshold = __shfl_sync(FULL_MASK, final, lane_id) ? 1 : MERGING_THRESHOLD;

    while (numPrim > threshold) {
        uint32_t nearest_neighbor = find_nearest_neighbor(numPrim, cluster_bounds);
        numPrim = merge_clusters(numPrim, nearest_neighbor, cluster_index, cluster_bounds, state);
    }

    store_indicies(num_left + num_right, cluster_index, state, left_start);
}

__global__ void build_bvh(build_state state, uint32_t *morton_codes){
    const uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;

    uint32_t left = index;
    uint32_t right = index;

    uint32_t split = 0;

    bool lane_active = index < state.prim_count;

    while(__ballot_sync(FULL_MASK, lane_active)){

        if(lane_active){

            uint32_t previous_id;

            if(find_parent_id(left, right, state.prim_count, morton_codes) == right){

                previous_id = atomicExch(&state.parent_indicies[right], left);

                if(previous_id != INVALID_IDX){
                    split = right + 1;

                    right = previous_id;
                }
            }
            else {

                previous_id = atomicExch(&state.parent_indicies[left - 1], right);

                if(previous_id != INVALID_IDX){
                    split = left;

                    left = previous_id;
                }
            }

            if(previous_id == INVALID_IDX){
                lane_active = false;
            }
        }

        uint32_t size = right - left + 1;
        bool final = lane_active && size == state.prim_count;

        uint32_t warp_mask = __ballot_sync(FULL_MASK, lane_active && (size > MERGING_THRESHOLD) || final);

        while (warp_mask){
            uint32_t lane_id = __ffs(warp_mask) - 1;

            ploc_merge(lane_id, left, right, split, final, state);

            warp_mask = warp_mask & (warp_mask - 1);
        }
    }
}

__device__ cluster make_cluster_from_leaf(int node_idx, float3 min, float3 max) {
    cluster c;
    c.node_idx = node_idx;
    c.min = min;
    c.max = max;
    c.active = 1;
    return c;
}

__global__ void make_clusters(
    unsigned int *sorted_object_ids,
    unsigned int num_objects,
    const float3 *positions,
    const int *pos_indices,
    bvh_node *nodes,
    cluster *clusters) {
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= num_objects)
        return;

    const unsigned int object_id = sorted_object_ids[thread_id];
    const int idx_u = pos_indices[3 * object_id + 0];
    const int idx_v = pos_indices[3 * object_id + 1];
    const int idx_w = pos_indices[3 * object_id + 2];

    const float3 u = positions[idx_u];
    const float3 v = positions[idx_v];
    const float3 w = positions[idx_w];
    const float3 min = fminf(u, fminf(v, w));
    const float3 max = fmaxf(u, fmaxf(v, w));

    const int leaf_node_idx = get_leaf_node_idx(thread_id, num_objects);
    bvh_node &leaf = nodes[leaf_node_idx];
    leaf.object_id = object_id;
    leaf.child_l = -1;
    leaf.child_r = -1;
    leaf.min = min;
    leaf.max = max;

    clusters[thread_id] = make_cluster_from_leaf(leaf_node_idx, min, max);

    if (thread_id < num_objects - 1)
        nodes[thread_id].paren = -1;
}

__device__ cluster merge_clusters_to_node(
    const cluster &left,
    const cluster &right,
    bvh_node *nodes,
    int new_node_idx) {
    cluster parent;
    parent.node_idx = new_node_idx;
    parent.min = fminf(left.min, right.min);
    parent.max = fmaxf(left.max, right.max);
    parent.active = 1;

    bvh_node &node = nodes[new_node_idx];
    node.child_l = left.node_idx;
    node.child_r = right.node_idx;
    node.paren = -1;
    node.min = parent.min;
    node.max = parent.max;
    node.visited = 0;

    nodes[left.node_idx].paren = new_node_idx;
    nodes[right.node_idx].paren = new_node_idx;

    return parent;
}

__device__ bool is_active_cluster(const cluster &c) {
    return c.active != 0 && c.node_idx != -1;
}

struct kernel_timer {
    cudaEvent_t start, stop;
    const char *name;

    kernel_timer(const char *n) : name(n) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~kernel_timer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void begin() { cudaEventRecord(start); }
    void end() { cudaEventRecord(stop); }
    float ms() {
        cudaEventSynchronize(stop);
        float t = 0.f;
        cudaEventElapsedTime(&t, start, stop);
        return t;
    }
};

bool build(const scene &s, bvh &bvh) {
    const int num_triangles = s.pos_indices.size() / 3;
    // must have at least two triangles. we cannot build a bvh for zero
    // triangles, and a bvh of one triangle has no internal nodes
    // which requires special handling which we forgo
    if (num_triangles <= 1) {
        fprintf(stderr, "too few triangles in scene: %d", num_triangles);
        return false;
    }

    // allocate array for morton and ids in dimension of triangles
    buf_gpu<unsigned int> d_morton;
    RETURN_IF_FALSE(d_morton.resize(num_triangles));
    buf_gpu<unsigned int> d_ids;
    RETURN_IF_FALSE(d_ids.resize(num_triangles));

    // sorted key, value pairs according to morton codes
    buf_gpu<unsigned int> d_morton_sorted;
    RETURN_IF_FALSE(d_morton_sorted.resize(num_triangles));
    buf_gpu<unsigned int> d_ids_sorted;
    RETURN_IF_FALSE(d_ids_sorted.resize(num_triangles));

    auto sort = [&](void *d_tmp, size_t &tmp_size) {
        // We don't actually need `keys_out` and `values_in` can be a
        // counting iterator, but `DeviceRadixSort` needs both as
        // backing storage.
        return cub::DeviceRadixSort::SortPairs(
            d_tmp,
            tmp_size,
            d_morton.get_ptr(),        // keys_in
            d_morton_sorted.get_ptr(), // keys_out
            d_ids.get_ptr(),           // values_in
            d_ids_sorted.get_ptr(),    // values_out
            num_triangles);
    };

    // Determine temporary device storage requirements.
    size_t num_tmp_bytes = 0;
    RETURN_IF_CUDA_ERR(sort(nullptr, num_tmp_bytes));
    // Allocate temporary storage
    buf_gpu<char> d_tmp;
    RETURN_IF_FALSE(d_tmp.resize(num_tmp_bytes));

    // copy scene to device
    RETURN_IF_FALSE(bvh.positions.resize(s.positions.size()));
    RETURN_IF_CUDA_ERR(cudaMemcpy(
        bvh.positions.get_ptr(),
        s.positions.data(),
        sizeof(float3) * s.positions.size(),
        cudaMemcpyHostToDevice));
    RETURN_IF_FALSE(bvh.normals.resize(s.normals.size()));
    RETURN_IF_CUDA_ERR(cudaMemcpy(
        bvh.normals.get_ptr(),
        s.normals.data(),
        sizeof(float3) * s.normals.size(),
        cudaMemcpyHostToDevice));
    RETURN_IF_FALSE(bvh.pos_indices.resize(s.pos_indices.size()));
    RETURN_IF_CUDA_ERR(cudaMemcpy(
        bvh.pos_indices.get_ptr(),
        s.pos_indices.data(),
        sizeof(int) * s.pos_indices.size(),
        cudaMemcpyHostToDevice));
    RETURN_IF_FALSE(bvh.nor_indices.resize(s.nor_indices.size()));
    RETURN_IF_CUDA_ERR(cudaMemcpy(
        bvh.nor_indices.get_ptr(),
        s.nor_indices.data(),
        sizeof(int) * s.nor_indices.size(),
        cudaMemcpyHostToDevice));

    // allocate BVH (n - 1 internal nodes, n leaf nodes)
    RETURN_IF_FALSE(bvh.nodes.resize(num_triangles - 1 + num_triangles));

    // events for measuring elapsed time
    cudaEvent_t start, stop;
    RETURN_IF_CUDA_ERR(cudaEventCreate(&start));
    RETURN_IF_CUDA_ERR(cudaEventCreate(&stop));
    RETURN_IF_CUDA_ERR(cudaEventRecord(start));

    const int block_size = 128;
    const int num_blocks = ceiling_div(num_triangles, static_cast<unsigned int>(block_size));

    kernel_timer t_morton("assign_morton");
    kernel_timer t_sort("radix_sort"); // Don't think we'll be messing with this one
    kernel_timer t_leaf("leaf_nodes");
    kernel_timer t_intrn("internal_nodes");
    kernel_timer t_aabb("set_aabb");

    t_morton.begin();
    assign_morton<<<num_blocks, block_size>>>(
        bvh.positions.get_ptr(),
        bvh.pos_indices.get_ptr(),
        s.soffset,
        s.sextent,
        d_morton.get_ptr(),
        d_ids.get_ptr(),
        num_triangles);
    t_morton.end();
    RETURN_IF_CUDA_ERR(cudaGetLastError());

    // Run sorting operation, sorting is stable.
    // https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceRadixSort.html
    t_sort.begin();
    RETURN_IF_CUDA_ERR(sort(d_tmp.get_ptr(), num_tmp_bytes));
    t_sort.end();

    build_state state;

    state.nodes = &bvh.nodes;
    state.cluster_indicies; //allocate space n stuff
    state.prim_count = num_triangles;
    state.cluster_count = 0;

    // construct leaf nodes
    t_leaf.begin();
    leaf_nodes<<<num_blocks, block_size>>>(
        d_ids_sorted.get_ptr(), num_triangles, bvh.nodes.get_ptr());
    t_leaf.end();
    RETURN_IF_CUDA_ERR(cudaGetLastError());

    // print elapsed time
    RETURN_IF_CUDA_ERR(cudaEventRecord(stop));
    RETURN_IF_CUDA_ERR(cudaEventSynchronize(stop));
    float milliseconds = 0.f;
    RETURN_IF_CUDA_ERR(cudaEventElapsedTime(&milliseconds, start, stop));
    const float seconds = milliseconds * 1e-3f;
    printf(
        "(fast) building took %6.5fms, %6.2f million triangles per second\n",
        milliseconds,
        num_triangles / seconds * 1e-6f);
    printf("  assign_morton:   %7.4f ms\n", t_morton.ms());
    printf("  radix_sort:      %7.4f ms\n", t_sort.ms());
    printf("  leaf_nodes:      %7.4f ms\n", t_leaf.ms());
    printf("  internal_nodes:  %7.4f ms\n", t_intrn.ms());
    printf("  set_aabb:        %7.4f ms\n", t_aabb.ms());
    return true;
}