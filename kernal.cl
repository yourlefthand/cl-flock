__kernel void knn(
	__global int8* starling,
	__global int* world,
	__global int8* out,
	int world_size_x, int world_size_y
	)
{
	unsigned int gid = get_global_id(0);

	private int3 separation;
	private int3 cohesion;
	private int3 alignment;

	private int2 p = starling[gid].s01;

	private int inner_rad = starling[gid].s2;
	private int outer_rad = starling[gid].s3;

	for(int i = (-1 * outer_rad);i <= outer_rad; i++){
		int y = p.y + i;
		int row = y * world_size_y;
		for(int k = (-1 * outer_rad); k <= outer_rad; k++){
			int x = p.x + k;
			int column = row + x;
			if (row >= 0 && row < world_size_y && column >= row && column < row + world_size_y){
				if (world[column] > 0){
					private int2 neighbor;
					neighbor.x = x + p.x;
					neighbor.y = y + p.y;

					float d = distance(convert_float2(p),convert_float2(neighbor));
					
					if(d <= outer_rad && d > inner_rad){

					}
					if(d <= inner_rad && d > 0){

					}
				}
				else {
					out[gid].s0 = 0;
					out[gid].s1 = 0;
				}
			}
		}	
	}
}