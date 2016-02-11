#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

const sampler_t world_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_REPEAT | CLK_FILTER_NEAREST;
//const sampler_t world_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

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

float f_cos_sim(
	int3 v_vec,
	int3 i_vec
	)
{
	private float result;
	
	private float3 origin = 0;

	private float3 f_i_vec = convert_float3(i_vec);
	private float3 f_v_vec = convert_float3(v_vec);

	private float idist = distance(origin, f_i_vec);
	private float vdist = distance(origin, f_v_vec);

	private float vdot = dot(f_i_vec, f_v_vec);

	return idist;
}

/*
 * given the dimension limitations in 3 dimensions (i.e. dimensions = [3,3,3])
 * we take an integer value and turn it into a normalized coordinate centered around the origin [0,0,0]
 * remember: denominator is the cum-product of the dimension limit [0:d) 
 */

int4 count(
	int3 dimensions,
	int value
	)
{
	private int4 coords = 0;
	coords.x = ((value / (1)) % dimensions.x) - (dimensions.x / 2);
	coords.y = ((value / (1 * dimensions.x)) % dimensions.y) - (dimensions.y / 2);
	coords.z = ((value / (1 * dimensions.x * dimensions.y)) % dimensions.z) - (dimensions.z / 2);
	return coords; 
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

	int3 origin = 0;

	int3 world_size;
	world_size.x = world_size_x;
	world_size.y = world_size_y;
	world_size.z = world_size_z;

	int3 window;
	window.x = min((2 * outer_rad) + 1, world_size.x);
	window.y = min((2 * outer_rad) + 1, world_size.y);
	window.z = min((2 * outer_rad) + 1, world_size.z);

	int max_vel = max(window.x, max(window.y, window.z)) / 15;

	int4 p = 0;
	p.s012 = starling[gid].s012;
	
	int4 v = 0;
	v.s012 = starling[gid].s345;
	
	int seen = 0;
	int coheded = 0;
	int separated = 0;

	float4 separation = 0;
	float4 cohesion = 0;

	int local_dot_debug;
	// int4 image_debug;
	// int4 local_pos_debug;
	// int4 world_pos_debug;

	for (int i = 0; i < (window.x * window.y * window.z); i++){
		int4 local_pos = count(window, i);

		int4 world_pos = p + local_pos;

		int4 image_read_out;
		image_read_out.x = world_pos.z;
		image_read_out.y = world_pos.y;
		image_read_out.z = world_pos.x;

		int4 image_read = read_imagei(world, world_sampler, image_read_out);

		// image_debug = image_read_out;
		// local_pos_debug = local_pos;
		// world_pos_debug = world_pos;

		/* 
		 * I'd like to get this to work - we could write the convolution layers out to images -> numpy arrays -> your-format-here
		 * but, i can't get 3d images to write! despite the above 'pragma' (nice). This can be returned to.	
		 * local_world = write_imagei(local_world, local_world_pos, CLK_UNSIGNED_INT8);
		 */

		if (image_read.x > 0) {

			float4 f_local_pos = convert_float(local_pos);

			float4 f_v = convert_float(v);

			float local_dist = length(f_local_pos);
			float v_dist = length(f_v);

			float local_dot = dot(f_local_pos, f_v) / (v_dist * local_dist);
			local_dot = min(local_dot, (float) 0.1);

			// local_dot_debug = convert_int_rtz(local_dot * 10);

			//float local_dist = f_distance(origin, local_pos.s012);
			//float local_sim = f_dot(local_pos.s012, v.s012);

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
					cohesion = cohesion + (f_local_pos * ((local_dist * 2) / outer_rad));
					coheded = coheded + 1;
				} else {
					// searched pont is within the outer zone of the target's perception
					separation = separation - (f_local_pos * ((inner_rad - local_dist) / inner_rad));
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
		}
	}

    
	int m = starling[gid].s6;
	int l = starling[gid].s7;

	float3 accel_frame = 0;
	int3 velocity = 0;
	int3 position = 0;

	//private int4 desire = p + convert_int((cohesion / coheded) - (separation / separated));
	
	float4 desire = (cohesion / (coheded + 1)) - (separation / (separated + 1));

	//float4 desire = cohesion - separation;

	/*
	 * there has to be a better way
     */

	int3 desire_debug = convert_int3_rtz(desire.s012);

	accel_frame = convert_float(l) / convert_float(m) * desire.s012;

	int3 accel_frame_debug = convert_int3_rtz(accel_frame);

	velocity = starling[gid].s345 + convert_int3_rtz(accel_frame.s012);

	position = p.s012 + velocity;

    // position = desire.s012;

	if (position.x < 0) {
		position.x = world_size.x + (position.x % world_size.x);
	}
	if (position.x > world_size.x - 1){
		position.x = position.x % world_size.x;
	}

	if (position.y < 0) {
		position.y = world_size.y + (position.y % world_size.y);
	}
	if (position.y > world_size.y - 1){
		position.y = position.y % world_size.y;;
	}

	if (position.z < 0) {
		position.z = world_size.z + (position.z % world_size.z);
	}
	if (position.z > world_size.z - 1){
		position.z = position.z % world_size.z;
	}

	out[gid].s012 = position;

	out[gid].s345 = velocity;

	out[gid].s6 = m;
	out[gid].s7 = l;
	out[gid].s8 = coheded;
	out[gid].s9 = separated;

	out[gid].sabcdef = 0;

	// out[gid].sab = convert_int3_rtz(cohesion.s01);
	// out[gid].sc = coheded;
	// out[gid].sde = convert_int3_rtz(separation.s01);
	// out[gid].sf = separated;


	// out[gid].sc = seen;
	// out[gid].sd = coheded;
	// out[gid].se = separated;

	// out[gid].sd = get_image_width(world);
	// out[gid].se = get_image_height(world);
	// out[gid].sf = get_image_depth(world);
}
