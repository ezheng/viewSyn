#version 420
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

uniform mat4 transform0;
uniform isampler2D depthTex0;
uniform float step;

out vec4 projTextureCoord;

void main()
{
	int depthIndex[3];

	for(int i = 0; i<3; i++)
	{ 
		vec2 texCoord = gl_in[i].gl_Position.xy;
		depthIndex[i] = texture(depthTex0, texCoord).x;
		
	}
	if( abs(depthIndex[0] - depthIndex[1])<3 && abs(depthIndex[0] - depthIndex[2])<3 && abs(depthIndex[1] - depthIndex[2])<3)
	{
		for( int i = 0; i<3; i++)
		{
			float depth = -1.0f + step * float(depthIndex[i]);
			gl_Position = transform0 * vec4(gl_in[i].gl_Position.xy *2.0f - 1.0f, depth, 1.0f);

			projTextureCoord = vec4(gl_in[i].gl_Position.xy, depth * 0.5f + 1.0f, 1.0f);
			EmitVertex();
		}
	}
	EndPrimitive();
}
