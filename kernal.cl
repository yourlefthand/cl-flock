#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

const sampler_t world_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_REPEAT | CLK_FILTER_NEAREST;
//const sampler_t world_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

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

	float4 f_vel = convert_float(v);

	float v_dist = length(f_vel);
	
	int seen = 0;
	int coheded = 0;
	int separated = 0;

	float4 separation = 0;
	float4 cohesion = 0;

	int local_dot_debug;

	for (int i = 0; i < (window.x * window.y * window.z); i++){
		int4 local_pos = count(window, i);

		int4 world_pos = p + local_pos;

		int4 image_read_out;
		image_read_out.x = world_pos.z;
		image_read_out.y = world_pos.y;
		image_read_out.z = world_pos.x;

		int4 image_read = read_imagei(world, world_sampler, image_read_out);

		float4 f_local_pos = convert_float(local_pos);

		float local_dist = length(f_local_pos);

		float cos_sim = dot(f_vel, local_dist) / (v_dist * local_dist);

		float inv_cos_sim = -1 * (1 + max(cos_sim, (float) 0));

		if (islessequal(local_dist, (float) outer_rad) != 0) {
			if (image_read.x > 0) {
				if (isgreater(local_dist, (float) inner_rad)) {
					cohesion = cohesion + (f_local_pos * inv_cos_sim); // * ((2 * local_dist) / outer_rad));
					coheded = coheded + 1;
				}
			} else {
				if (islessequal(local_dist, (float) inner_rad)) {
					separation = separation + (f_local_pos * inv_cos_sim); // * ((2 * (outer_rad - local_dist)) / outer_rad));
					separated = separated + 1;
				}
			}
			seen = seen + 1;
		}
	}

	float l = convert_float(starling[gid].s6);
	float m = convert_float(starling[gid].s7);

	float acc_coef = l / m;

	float3 accel_frame = 0;
	int3 velocity = 0;
	int3 position = 0;
	
	float4 normal_cohede = (cohesion / max(coheded, 1));
	float4 normal_separate = (separation / max(separated, 1));

	float4 desire = ((normal_cohede) + (normal_separate));

	int3 desire_debug = convert_int3_rtz(desire.s012);

	float4 impulse = -1 * (f_vel - desire);

	int3 impulse_debug = convert_int3_rtz(impulse.s012);

	accel_frame = acc_coef * impulse.s012;

	int3 accel_frame_debug = convert_int3_rtz(accel_frame);

	velocity = v.s012 + convert_int3_rtz(accel_frame.s012);	

	position = p.s012 + velocity;

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

	out[gid].s6 = l;
	out[gid].s7 = m;
	out[gid].s8 = coheded;
	out[gid].s9 = separated;

	out[gid].sabc = desire_debug.s012;
	out[gid].sdef = accel_frame_debug.s012;
}
