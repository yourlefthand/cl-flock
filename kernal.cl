

__kernel void flock(
	__global int4* pos,
	__global int* mass,
	__global int* power,
	__global int* lifetime,
	__global int* world,
	__global int4* out,
	int inner_rad, int outer_rad,
	int world_size_x, int world_size_y
	)
{
	unsigned int gid = get_global_id(0);

	// int4 collector;

	// int2 goal;

	// float d;

	// int2 p = pos[gid].lo;
	// int2 v = pos[gid].hi;

	// int j = power[gid];
	// int l = lifetime[gid];
	// int m = mass[gid];

	// uint n_cohere;
	// uint2 p_cohere;

	// uint n_separate;
	// uint2 p_separate;

	// for(int i = (-1 * outer_rad);i <= outer_rad; i++){
	// 	int y = p.y + i;
	// 	int row = y * world_size_y;
	// 	for(int k = (-1 * outer_rad); k <= outer_rad; k++){
	// 		int x = p.x + k;
	// 		int column = row + x;
	// 		if (world[column] > 0){
	// 		}
	// 	}
	// }

//	collector.z = gid;
	out[gid].x = 0;
	out[gid].y = 0;
}