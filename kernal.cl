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
					out[gid].s0 = convert_int_rte(d);
					if (islessequal(d, convert_float(outer_rad)) > 0){
						if (isgreater(d, convert_float(inner_rad)) > 0){
							cohesion.z++;
							cohesion.lo += (p - neighbor) % convert_int_sat(outer_rad - d);
						}
						else{
							separation.z++;
							separation.lo += -1 * (p - neighbor) * convert_int_sat(inner_rad - d); 
						}
					}
					else{
						out[gid].s0 = p.x;
						out[gid].s1 = p.y;
					}
				}
				else {
					out[gid].s0 = p.x;
					out[gid].s1 = p.y;
				}
			}
			else {
				out[gid].s0 = p.x;
				out[gid].s1 = p.y;
			}
		}
	}

	private int4 intention;
	intention.lo = (cohesion.lo / cohesion.z);
	intention.hi = (separation.lo / separation.z);

	out[gid].s01 = p;
}