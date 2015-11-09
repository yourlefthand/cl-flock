#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

const sampler_t world_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

int4 resolve_difference(
	int4 a_vec,
	int4 b_vec
	)
{
 	private int4 difference_vector = a_vec - b_vec;

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
	__global int16* starling,
	__global int16* out,
	image3d_t world,
	int world_size_x, int world_size_y, int world_size_z,
	int inner_rad, int outer_rad
	)
{	
	unsigned int gid = get_global_id(0);

	private int search_dimension = ((2 * outer_rad) + 1);

	private int* local_world[125];

//	private int3 alignment = 0;

	private int4 p;
	p.s012 = starling[gid].s012;
	private int4 v;
	v.s012 = starling[gid].s345;

	private int seen = 0;
	private int coheded = 1;
	private int separated = 1;

	private int4 direction = resolve_difference(p, v);
	private float4 separation = 0;
	private float4 cohesion = 0;


	for(int i = (-1 * outer_rad);i <= outer_rad; i++){
		int z = (p.z + i) % world_size_z;
		int layer = z * world_size_y * world_size_x;
		for(int k = (-1 * outer_rad); k <= outer_rad; k++){
			int y = (p.y + k) % world_size_y;
			int row = y * world_size_x;	
			for(int j = (-1 * outer_rad); j <= outer_rad; j++){
				int x = (p.x + j) % world_size_x;
				int column = x;

			//	int world_pos = layer + row + column;
				int4 world_pos;
				world_pos.x = x;
				world_pos.y = y;
				world_pos.z = z;

				int4 image_read_out;
				image_read_out.x = z;
				image_read_out.y = y;
				image_read_out.z = x;

				int4 local_world_pos = 0; 
				local_world_pos.x = j;
				local_world_pos.y = k;
				local_world_pos.z = i;

				//it's an RGBA color object dummy! I was overwriting it this whole time because of some stupid assumption I made...duh!
				int4 image_read = read_imagei(world, world_sampler, image_read_out);

			//	local_world[local_world_pos.x + (local_world_pos.y * search_dimension) + (local_world_pos.z * search_dimension * search_dimension)] = local_world_pos.w; 		


			/* 
			 * I'd like to get this to work - we could write the convolution layers out to images -> numpy arrays -> your-format-here
			 * but, i can't get 3d images to write! despite the above 'pragma' (nice). This can be returned to.	
			 * local_world = write_imagei(local_world, local_world_pos, CLK_UNSIGNED_INT8);
			 */

			 // it seems that there is some plane of values that we can catch, otherwise we miss
			 // 

				if (image_read.x > 0){

					//if (x != 0 && y !=0 && z != 0){
						// if (x >= 0 && x < world_size_x){
						// 	if (y >= 0 && y < world_size_y){
						// 		if (z >= 0 && z < world_size_z) {
									int4 difference_vector = resolve_difference(world_pos, p);
									float4 f_difference_vector = convert_float(difference_vector);
					
									float local_dist = f_distance(p.s012, world_pos.s012);
									float local_dot = f_dot(direction.s012, difference_vector.s012);

									int local_dot_i = convert_int(local_dot);

									/* 
									 * here we have 'cohesion and separation' conditions
									 * it would be nice to translate this into a single function --- hm, a sigmoid?!
									 * if I could add signal passing to this, I'd have a neural layer!!! not only that, but a layer
									 * for analyzing matrices of data **convolutionally** (i.e. framed)
									 * it'd be interesting to return to the perceptron capabilities in that Minsky & Papert i.e.
									 * optimizing search pattern - check this out: http://danielrapp.github.io/morph/ for generative
									 * lensing!
									 * 
									 * turning this switch into a lambda with emittable values will make all the difference.
									 * at the moment- perhaps this will change - we will want to keep 'moving' the target value a la
									 * simulation - this data stream will be passed to a time-dilation layer - genetic memory
									 */
									if (islessequal(local_dist, (float) outer_rad) != 0) {
										if (isgreater(local_dist, (float) inner_rad) != 0) {
											// searched point is in that outer zones of the target's perception
											cohesion = cohesion + f_difference_vector;
											coheded = coheded + 1;
										} else {
											// searched pont is within the outer zone of the target's perception
											separation = separation + f_difference_vector;
											separated = separated + 1;
										}	
									}

									seen = seen + 1;

								/* 
								 * what I need here is an algorithm that can do the following: make an assumption about the direction
								 * of a neighbor by comparing the hash of it's velocity (vector) to their hashes via the color channel
								 * of the received image. This is complicated by the fact that I have no idea how to store that info
								 * in those goddamn image objects and that the hash is so simple it will result in three 'phases' during
								 * comparison - such that our given comparitor vector is simply rotated and hashed - appearing 
								 * identical, despite being somehor orthogonal. Still, understanding the period of these phsases
								 * would allow us to predict those that are facing (thus threatening to collide with) our unit - with
								 * some modular accuracy. we'll use this 'alignment' calculation to weigh our intended escape vector
								 * and subsequent power applications. translated into a vector (after casting our die, so to speak)
								 * we can now calculate position forward one frame, and proceed to update the model
								 */


						// 		}
						// 	}
						// }
					//}
				}
			}
		}
	}
    
	int m = starling[gid].s6;
	int l = starling[gid].s7;

	float3 accel_frame = 0;
	int3 velocity = 0;
	int3 position = 0;

	//private int4 desire = p + convert_int((cohesion / coheded) - (separation / separated));
	float4 desire = (cohesion / coheded) - (separation / separated);

	/*
	 * there has to be a better way
     */

	int3 desire_debug = convert_int_rtz(desire.s012);

	accel_frame = convert_float(l) / convert_float(m) * desire.s012;

	int3 accel_frame_debug = convert_int_rtz(accel_frame);

	velocity = (starling[gid].s345 + convert_int_rtz(accel_frame.s012)) % (world_size_x / 2);

	position = p.s012 + velocity;

    // position = desire.s012;

	if (position.x < 0) {
		position.x = world_size_x + (position.x % world_size_x);
	}
	if (position.x > world_size_x - 1){
		position.x = position.x % world_size_x;
	}

	if (position.y < 0) {
		position.y = world_size_y + (position.y % world_size_y);
	}
	if (position.y > world_size_y - 1){
		position.y = position.y % world_size_y;;
	}

	if (position.z < 0) {
		position.z = world_size_z + (position.z % world_size_z);
	}
	if (position.z > world_size_z - 1){
		position.z = position.z % world_size_z;
	}

	out[gid].s012 = position;

	out[gid].s345 = velocity;

	out[gid].s6 = m;
	out[gid].s7 = l;
	out[gid].s8 = 0;

	out[gid].s9ab = accel_frame_debug;
	out[gid].scde = desire_debug;
	// out[gid].s9ab = convert_int(separation.s012);
	// out[gid].sc = seen;
	// out[gid].sd = coheded;
	// out[gid].se = separated;

	// out[gid].sc = get_image_width(world);
	// out[gid].sd = get_image_height(world);
	// out[gid].se = get_image_depth(world);
	
	seen = 0;
	coheded = 0;
	separated = 0;
}
