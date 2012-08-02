#version 420
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

uniform mat4 transform0;
uniform mat4 transform1;
uniform usampler2D depthTex0;
uniform usampler2D depthTex1;

uniform float step;

out vec4 projTextureCoord;


void main()
{
	//float xy[6] = {-0.5, -0.5, -0.5, 0.5, 0.5, 0.7};
	
	for(int i = 0; i<3; i++)
	{ 
		if(gl_InvocationID == 0)
		{
			vec2 texCoord = gl_in[i].gl_Position.xy;
			uint depthIndex = texture(depthTex0, texCoord).x;
			float depth = -1.0f + step * float( depthIndex );

			gl_Position = transform0 * vec4(gl_in[i].gl_Position.xy *2.0f - 1.0f, depth, 1.0f);
			projTextureCoord = vec4(texCoord, depth * 0.5f + 1.0f, 1.0f);
			EmitVertex();
		}
	//--------------------------------------------------------------------------------------------
	/*	else if(gl_InvocationID == 2);
		{
			vec2 texCoord1 = gl_in[i].gl_Position.xy;
			uint depthIndex1 = texture(depthTex1, texCoord1).x;
			float depth1 = -1.0f + step * float( depthIndex1 );

			gl_Position = transform1 * vec4(gl_in[i].gl_Position.xy *2.0f - 1.0f, depth1, 1.0f);
			projTextureCoord = vec4(texCoord1, depth1 * 0.5f + 1.0f, 1.0f);
			EmitVertex();
		}*/
	}
	EndPrimitive();
}
