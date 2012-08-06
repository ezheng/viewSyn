#version 420
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

uniform mat4 transform0;
uniform usampler2D depthTex0;
uniform float step;
in int instanceID[];

out vec4 projTextureCoord;
out int theInstaceID;

void main()
{
	int depthIndex[3];

	for(int i = 0; i<3; i++)
	{ 
		vec2 texCoord = gl_in[i].gl_Position.xy;

		if(instanceID[0] == 0)
			depthIndex[i] = int(texture(depthTex0, texCoord).x);
		else if(instanceID[0] == 1);
			depthIndex[i] = int(texture(depthTex1, texCoord).x);
		
	}
	if( abs(depthIndex[0] - depthIndex[1])<13 && abs(depthIndex[0] - depthIndex[2])<13 && abs(depthIndex[1] - depthIndex[2])<13)
	{
		for( int i = 0; i<3; i++)
		{
			float depth = -1.0f + step * float(depthIndex[i]);
			if(instanceID[0] == 0)
				gl_Position = transform0 * vec4(gl_in[i].gl_Position.xy *2.0f - 1.0f, depth, 1.0f);
			else if (instanceID[0] == 1)
				gl_Position = transform1 * vec4(gl_in[i].gl_Position.xy *2.0f - 1.0f, depth, 1.0f);

			projTextureCoord = vec4(gl_in[i].gl_Position.xy, depth * 0.5f + 1.0f, 1.0f);
			gl_Layer = instanceID[0];
			theInstaceID = instanceID[0];
			EmitVertex();
		}
	}

	EndPrimitive();
	
}
