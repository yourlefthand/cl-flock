#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

int4 resolve_difference(
	int8 metrics
	)
{
 	private int4 difference_vector;
	int3 vector = metrics.s012 - metrics.s345;
	difference_vector.s012 = vector;

	//difference_vector.w = hashed vector direction base 255

	return difference_vector;
}

float f_distance(
	int3 a_pos,
	int3 b_pos
	)
{
	private float result;
	
	private float3 a_pos_f = convert_float3(a_pos);
	private float3 b_pos_f = convert_float3(b_pos);

	result = distance(a_pos_f, b_pos_f);

	return result;
}

float f_dot(
		int3 a_vec,
		int3 b_vec
		)
{
	private float result;

	private float3 a_vec_f = convert_float3(a_vec);
	private float3 b_vec_f = convert_float3(b_vec);

	result = dot(a_vec_f, b_vec_f);

	return result;
}

int3 check_border(
		int world_size_x,
		int world_size_y,
		int world_size_z,
		int3 intention
		)
{
	private int3 outcome;

	private int3 world;
	world.x = world_size_x;
	world.y = world_size_y;
	world.z = world_size_z;

	if (intention.x < 0) {
		outcome.x = world.x + intention.x;
	}

	if (intention.x >= world.x){
		outcome.x = 0 + (intention.x - world.x);
	}

	if (intention.y < 0) {
		outcome.y = world.y + intention.y;
	}

	if (intention.y >= world.y){
		outcome.y = 0 + (intention.y - world.y);
	}

	if (intention.z < 0) {
		outcome.z = world.z + intention.z;
	}

	if (intention.z >= world.z){
		outcome.z = 0 + (intention.z - world.z);
	}

	return outcome;
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

	private int4 intention = 0;	

	private int3 separation = 0;
	private int3 cohesion = 0;
//	private int3 alignment = 0;

	private int3 p = starling[gid].s012;
	private int3 v = starling[gid].s345;

	private int m = starling[gid].s6;
	private int w = starling[gid].s7;

	private int proxy = 0;

	private int4 direction = resolve_difference(starling[gid]);


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

				int4 local_world_pos = read_imagei(world, world_sampler, world_pos); 
				local_world_pos.x = j;
				local_world_pos.y = k;
				local_world_pos.z = i;

			//	local_world[local_world_pos.x + (local_world_pos.y * search_dimension) + (local_world_pos.z * search_dimension * search_dimension)] = local_world_pos.w; 		


			/* I'd like to get this to work - we could write the convolution layers out to images -> numpy arrays -> your-format-here
			 * but, i can't get 3d images to write! despite the above 'pragma' (nice). This can be returned to.	
				local_world = write_imagei(local_world, local_world_pos, CLK_UNSIGNED_INT8);
			*/


				if (local_world_pos.w > 0){
					if (x >= 0 && x < world_size_x){
						if (y >= 0 && y < world_size_y){
							if (z >= 0 && z < world_size_z) {
								int8 comparison_vector;
								comparison_vector.lo = starling[gid].lo;
								comparison_vector.hi = world_pos;
								
								int4 difference_vector = resolve_difference(comparison_vector);
				
								float local_dist = f_distance(comparison_vector.s012, world_pos.s012);
								float local_dot = f_dot(direction.s012, difference_vector.s012);

								int local_dot_i = convert_int(local_dot);

								// here we have 'cohesion and separation' conditions
								// it would be nice to translate this into a single function --- hm, a sigmoid?!
								// if I could add signal passing to this, I'd have a neural layer!!! not only that, but a layer
								// for analyzing matrices of data **convolutionally** (i.e. framed)
								// it'd be interesting to return to the perceptron capabilities in that Minsky & Papert i.e.
								// optimizing search pattern - check this out: http://danielrapp.github.io/morph/ for generative
								// lensing!
								//
								// turning this switch into a lambda with emittable values will make all the difference.
								// at the moment- perhaps this will change - we will want to keep 'moving' the target value a la
								// simulation - this data stream will be passed to a time-dilation layer - genetic memory
								if (islessequal(local_dist, (float) outer_rad) != 0) {
									if (isgreater(local_dist, (float) inner_rad) != 0) {
										// searched point is in that outer zones of the target's perception
										intention.w = intention.w + 1;
										intention = intention + local_world_pos;
									} else {
										// searched pont is within the outer zone of the target's perception
									}	
								}

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
//	out[gid].s012 = check_border(world_size_x, world_size_y, world_size_z, (p + (intention.s012 / intention.s3)));
	private int3 desire = p + proxy;
	//there has to be a better way
	if (desire.x < 0) {
		desire.x = world_size_x + desire.x;
	}
	if (desire.x > world_size_x - 1){
		desire.x = desire.x % world_size_x;
	}

	if (desire.y < 0) {
		desire.y = world_size_y + desire.y;
	}
	if (desire.y > world_size_y - 1){
		desire.y = desire.y % world_size_y;;
	}

	if (desire.z < 0) {
		desire.z = world_size_z + desire.z ;
	}
	if (desire.z > world_size_z - 1){
		desire.z = desire.z % world_size_z;
	}

	out[gid].s012 = desire;
	out[gid].s345 = p;
	out[gid].s6 = proxy;
	proxy = 0;
}
