#version 420
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;
uniform mat4x4 transformMatrix[2];
uniform float step;
in int instanceID[];
out vec4 ProjTexCoord[2];
out vec2 tex2dCoord;

void main() 
{
	float depth = -1.0f + step * float( instanceID[0] + 1);	
	float length  = 1.0f;


	gl_Position = gl_in[0].gl_Position + vec4( -length, -length, depth , 0.0f);
	for(int i = 0; i< transformMatrix.length; i++) 
	{	
		ProjTexCoord[i] = transformMatrix[i] * gl_Position;
	}
	gl_Layer = instanceID[0];
	tex2dCoord = vec2(0,0);
	EmitVertex();

	gl_Position = gl_in[0].gl_Position + vec4( -length, length, depth , 0.0f);
	for(int i = 0; i< transformMatrix.length; i++) 
	{
		ProjTexCoord[i] = transformMatrix[i] * gl_Position;
	}
	gl_Layer = instanceID[0];
	tex2dCoord = vec2(0,1);
	EmitVertex();
	
	gl_Position = gl_in[0].gl_Position + vec4( length, -length, depth , 0.0f);
	for(int i = 0; i< transformMatrix.length; i++) 
	{
		ProjTexCoord[i] = transformMatrix[i] * gl_Position;
	}
	gl_Layer = instanceID[0];
	tex2dCoord = vec2(1,0);
	EmitVertex();

	gl_Position = gl_in[0].gl_Position + vec4( length, length, depth , 0.0f);
	for(int i = 0; i< transformMatrix.length; i++) 
	{
		ProjTexCoord[i] = transformMatrix[i] * gl_Position;
	}
	gl_Layer = instanceID[0];
	tex2dCoord = vec2(1,1);
	EmitVertex();
	EndPrimitive();
}
