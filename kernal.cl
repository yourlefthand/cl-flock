#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

const sampler_t world_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_REPEAT | CLK_FILTER_NEAREST;
//const sampler_t world_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

/*
 * given the dimension limitations in 3 dimensions (i.e. dimensions = [3,3,3])
 * we take an integer value and turn it into a normalized coordinate centered around the origin [0,0,0]
 * remember: denominator is the cum-product of the dimension limit [0:d) 
 */

float3 grav_accel(
	float3 a_pos, 
	float a_mass, 
	float3 b_pos, 
	float b_mass
	)
{
	const float GRAVITATION = 1.0f;
	float3 accel = 0;

	float3 delta_pos = a_pos - b_pos;
	float dist = length(delta_pos);
	float dist_sq = pow(dist, 2.0f);

	float force = GRAVITATION * a_mass * b_mass / dist_sq;
	accel = force * delta_pos / dist;
	return accel;
}

float8 rk_accel(
	float3 a_pos, float3 a_vel, float a_mass,
	float3 b_pos, float b_mass,
	float3 c_pos, float c_mass
	)
{
	float8 out = 0;

	float3 origin = 0;
	
	float3 inv_vel = origin - a_vel;

	float3 k1 = grav_accel(a_pos, a_mass, b_pos, b_mass) 
		+ grav_accel(a_pos, a_mass, c_pos, c_mass);
		// + grav_accel(a_pos, a_mass, inv_vel, a_mass);

	float3 p2 = a_pos + a_vel;
	float3 v2 = a_vel + k1;
	float3 inv_v2 = p2 - v2;

	float3 k2 = grav_accel(p2, a_mass, b_pos, b_mass) 
		+ grav_accel(p2, a_mass, c_pos, c_mass);
		// + grav_accel(p2, a_mass, inv_v2, a_mass);

	float3 p3 = p2 + v2;
	float3 v3 = v2 + k2;
	float3 inv_v3 = p3 - v3;

	float3 k3 = grav_accel(p3, a_mass, b_pos, b_mass) 
		+ grav_accel(p3, a_mass, c_pos, c_mass);
		// + grav_accel(p3, a_mass, inv_v3, a_mass);

	float3 p4 = p3 + v3;
	float3 v4 = v3 + k3;
	float3 inv_v4 = p4 - v4;

	float3 k4 = grav_accel(p4, a_mass, b_pos, b_mass) 
		+ grav_accel(p4, a_mass, c_pos, c_mass);
		// + grav_accel(p4, a_mass, inv_v4, a_mass);

	float3 da = 1.0f/6.0f * (k1 + 2 * k2 + 2 * k3 + k4);
	float3 dv = 1.0f/6.0f * (a_vel + 2 * v2 + 2 * v3 + v4);
	float3 dp = a_pos + dv;

	out.s012 = da;
	out.s345 = dv;
	out.s67 = 0;
	return out;
}

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

	float4 f_pos = convert_float(p);
	
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

		//float inv_cos_sim = 1 - max(cos_sim, (float) 0.0);
		//float inv_cos_sim = 1;

		if (islessequal(local_dist, (float) outer_rad) != 0) {
			if (image_read.x > 0) {
				//if (isgreater(local_dist, (float) inner_rad)) {
					cohesion = cohesion + f_local_pos; // * (local_dist / outer_rad));
					coheded = coheded + 1;
				//}
			} else {
				//if (isgreater(local_dist, (float) inner_rad)) {
					separation = separation + f_local_pos; //* ((inner_rad - local_dist) / inner_rad));
					separated = separated + 1;
				//}
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

	float f_cohede = (float) coheded;
	float cohede_mass = f_cohede / pow(outer_rad, 2.0f) * m * 10;

	float f_separate = (float) separated;
	float separate_mass = f_separate / pow(outer_rad, 2.0f) * m * 10;

	float4 normal_cohede = (cohesion / max(coheded, 1));
	float4 normal_separate = (separation / max(separated, 1));

	float4 nn_cohede = normal_cohede * 116 / length(normal_cohede);
	float4 nn_separate = normal_separate * 116 / length(normal_separate);

	position = p.s012 + v.s012;

	float8 rk = rk_accel(f_vel.s012, 
						 f_vel.s012, 
						 m, 
						 nn_cohede.s012, 
						 cohede_mass,
						 nn_separate.s012,
						 separate_mass);

	velocity = convert_int3_rtz(rk.s012);

	int3 vel_debug = convert_int3_rtz(rk.s012);

	int3 pos_debug = convert_int3_rtz(rk.s345);

	// position = p.s012 + v.s012;

	// float4 desire = f_vel - ((normal_cohede) + (normal_separate));

	// int3 desire_debug = convert_int3_rtz(desire.s012);

	// accel_frame = acc_coef * desire.s012;

	// int3 accel_frame_debug = convert_int3_rtz(accel_frame);

	// velocity = v.s012 + convert_int3_rtz(accel_frame.s012);

	// if (abs(velocity.x) > max_vel){
	// 	velocity.x = 0;
	// }
	// if (abs(velocity.y) > max_vel){
	// 	velocity.y = 0;
	// }
	// if (abs(velocity.z) > max_vel){
	// 	velocity.z = 0;
	// }

    // position = p.s012 + velocity;

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

	//out[gid].sabcdef = 0;
	out[gid].sab = convert_int3_rtz(vel_debug.s01 * 10);
	out[gid].scd = convert_int3_rtz(pos_debug.s01 * 10);
	out[gid].sef = 0;
}
