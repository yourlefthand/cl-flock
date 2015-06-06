

__kernel void knn(
	__global int4* pos,
	__global int* mass,
	__global int* power,
	__global int* lifetime,
	__global int4* out,
	int inner_rad, int outer_rad
	)
{
	unsigned int gid = get_global_id(0);

	 int4 collector;

	 int2 goal;

	 float d;

	 int2 p = pos[gid].lo;
	 int2 v = pos[gid].hi;

	 int j = power[gid];
	 int l = lifetime[gid];
     int m = mass[gid];

     uint n_cohere;
     uint2 p_cohere;

     uint n_separate;
     uint2 p_separate;

	for(int i=0;i < get_global_size(0); i++){
		n_separate = i;
	}

	//collector.x = n_separate;
	//collector.hi = p_cohere;
	//collector.y += 1;
	out[gid].x = n_separate;
}