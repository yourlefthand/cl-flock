#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

int4 resolve_direction(
	int8 metrics
	)
{
 	private int4 hashed_vector;
	int3 vector = metrics.s012 - metrics.s345;
	hashed_vector.s012 = vector;
	hashed_vector.w = (vector.x * vector.y * vector.z);

	return hashed_vector;
}

__kernel void knn(
	__global int8* starling,
	__global int8* out,
	image3d_t world,
	int world_size_x, int world_size_y, int world_size_z,
	int inner_rad, int outer_rad
	)
{
	const sampler_t world_sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;;
	
	unsigned int gid = get_global_id(0);

	private int search_dimension = ((2 * outer_rad) + 1);

	private int* local_world[125];

	private int3 separation = 0;
	private int3 cohesion = 0;
//	private int3 alignment = 0;

	private int3 p = starling[gid].s012;
	private int3 v = starling[gid].s345;

	private int m = starling[gid].s6;
	private int w = starling[gid].s7;

	private int proxy = 0;

	private int4 vector = resolve_direction(starling[gid]);


	for(int i = (-1 * outer_rad);i <= outer_rad; i++){
		int z = p.z + i;
		int layer = z * world_size_y * world_size_x;
		for(int k = (-1 * outer_rad); k <= outer_rad; k++){
			int y = p.y + k;
			int row = y * world_size_x;	
			for(int j = (-1 * outer_rad); j <= outer_rad; j++){
				int x = p.x + j;
				int column = x;

			//	int world_pos = layer + row + column;
				int4 world_pos;
				world_pos.x = x;
				world_pos.y = y;
				world_pos.z = z;

				int4 local_world_val = read_imagei(world, world_sampler, world_pos); 
				local_world_val.x = j;
				local_world_val.y = k;
				local_world_val.z = i;

				local_world[local_world_val.x + (local_world_val.y * search_dimension) + (local_world_val.z * search_dimension * search_dimension)] = local_world_val.w; 		


			/* I'd like to get this to work - we could write the convolution layers out to images -> numpy arrays -> your-format-here
			 * but, i can't get 3d images to write! despite the above 'pragma' (nice). This can be returned to.	
				local_world = write_imagei(local_world, local_world_val, CLK_UNSIGNED_INT8);
			*/


				if (local_world_val.w > 0){
					if (x >= 0 && x < world_size_x){
						if (y >= 0 && y < world_size_y){
							if (z >= 0 && z < world_size_z) {
								int8 comparison_vector;
								comparison_vector.lo = starling[gid].lo;
								comparison_vector.hi = local_world_val;
								
								int4 result_vector = resolve_direction(comparison_vector);
								proxy = proxy + 1;
							/* what I need here is an algorithm that can do the following: make an assumption about the direction
							 * of a neighbor by comparing the hash of it's velocity (vector) to their hashes via the color channel
							 * of the received image. This is complicated by the fact that I have no idea how to store that info
							 * in those goddamn image objects and that the hash is so simple it will result in three 'phases' during
							 * comparison - such that our given comparitor vector is simply rotated and hashed - appearing 
							 * identical, despite being somehor orthogonal. Still, understanding the period of these phsases
							 * would allow us to predict those that are facing (thus threatening to collide with) our unit - with
							 * some modular accuracy. we'll use this 'alignment' calculation to weigh our intended escape vector
							 * and subsequent power applications. translated into a vector (after casting our die, so to speak)
							 * we can now calculate position forward one frame, and proceed to update the model
							 *
							 */


							}
						}
					}
				}
			}
		}
	}
	out[gid].s012 = p;
	out[gid].s345 = v;
	out[gid].s6 = proxy;
	proxy = 0;
}
