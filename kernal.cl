__kernel void knn(
	__global int8* starling,
	__global int* world,
	__global int8* out,
	__private int2 constants,
	int world_size_x, int world_size_y,
	int inner_rad, int outer_rad
	)
{
	unsigned int gid = get_global_id(0);

	private int3 separation = 0;
	private int3 cohesion = 0;
//	private int3 alignment = 0;

	private int2 p = starling[gid].s01;
	private int2 v = starling[gid].s23;

	private int m = starling[gid].s2;
	private int w = starling[gid].s3;

	for(int i = (-1 * outer_rad);i <= outer_rad; i++){
		int y = p.y + i;
		int row = y * world_size_y;
		for(int k = (-1 * outer_rad); k <= outer_rad; k++){
			int x = p.x + k;
			int column = row + x;
			if (row >= 0 && row < world_size_y && column >= row && column < row + world_size_y){
				if (world[column] > 0){
					private int2 neighbor = 0;
					neighbor.x = x + p.x;
					neighbor.y = y + p.y;

					float d = distance(convert_float2(p),convert_float2(neighbor));
					out[gid].s0 = convert_int_rte(d);
					if (islessequal(d, convert_float(outer_rad)) > 0){
						if (isgreater(d, convert_float(inner_rad)) > 0){
							cohesion.z++;
							//cohesion.lo += (neighbor - p) / convert_int_sat(outer_rad - d);
							cohesion.lo += 2 * (neighbor - p);
							// cohesion.lo += 0;
						}
						else{
							separation.z++;
							//separation.lo += -1 * (neighbor - p) * convert_int_sat(inner_rad - d);
							separation.lo += -1 * (neighbor - p); 
							// separation.lo += 0; 
						}
					}
					else{
						// out[gid].s0 = p.x;
						// out[gid].s1 = p.y;
					}
				}
				else {
					// out[gid].s0 = p.x;
					// out[gid].s1 = p.y;
				}
			}
			else {
				// out[gid].s0 = p.x;
				// out[gid].s1 = p.y;
			}
		}
	}

	private int2 intention;
	intention = (((cohesion.lo) + (separation.lo)) / 2);
	//intention = (cohesion.lo / cohesion.z);

	int2 a = 0;

	if (v.x > 0 || v.y > 0){
		if (v.x > 0){
				a.x = (w * intention.x) / (m * v.x);
		}
		if (v.y > 0){
				a.y = (w * intention.y) / (m * v.y); 
		}
	} else {
		a = (w * intention) / m;
	}

	v = v + a;

	if ((p.x + v.x) >= 0 && (p.x + v.x) < world_size_x && (p.y + v.y) >= 0 && (p.y + v.y) < world_size_y){
		out[gid].s01 = p + v;
		out[gid].s23 = v;
	} else{
		out[gid].s01 = p;
	}

	out[gid].s45 = intention;
	out[gid].s67 = 0;
}
