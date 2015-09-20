__kernel void knn(
	__global int8* starling,
	__global int8* out,
	image3d_t world,
	__private int2 constants,
	int world_size_x, int world_size_y, int world_size_z,
	int inner_rad, int outer_rad
	)
{
	unsigned int gid = get_global_id(0);

	private int3 separation = 0;
	private int3 cohesion = 0;
//	private int3 alignment = 0;

	private int3 p = starling[gid].s012;
	private int3 v = starling[gid].s345;

	private int m = starling[gid].s6;
	private int w = starling[gid].s7;

	private int proxy = 0;
/*
	for(int i = (-1 * outer_rad);i <= outer_rad; i++){
		int z = p.z + i;
		int layer = z * world_size_y * world_size_x;
		for(int k = (-1 * outer_rad); k <= outer_rad; k++){
			int y = p.y + k;
			int row = y * world_size_x;	
			for(int j = (-1 * outer_rad); j <= outer_rad; j++){
				int x = p.x + j;
				int column = x;

				int world_pos = layer + row + column;
				//if (world[world_pos]){
					if (x >= 0 && x < world_size_x){
						if (y >= 0 && y < world_size_y){
							if (z >= 0 && z < world_size_z) {
								out[gid].s0 = x;
								out[gid].s1 = y;
								out[gid].s2 = z;
								proxy = proxy + 1;
							}
						}
					}
				//}
			}
		}
	}
*/
	out[gid].s0 = 1234;
}
