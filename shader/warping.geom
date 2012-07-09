#version 420

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

uniform mat4x4 transformMatrix0;	// TexMatrix * ProjMatrix * ModelViewMatrix * virtualInvModelView * virtualInvProj
uniform mat4x4 transformMatrix1; 
uniform mat4x4 transformMatrix2;
uniform float step;

in int instanceID[];



out vec4 ProjTexCoord[3];



void main()
{
	
	float depth = -1.0f + step * float( instanceID[0] + 1) ;
	float length  = 1.0f;


	gl_Position = gl_in[0].gl_Position + vec4( -length, -length, depth , 0.0f);
	ProjTexCoord[0] = transformMatrix0 * gl_Position;
	ProjTexCoord[1] = transformMatrix1 * gl_Position;
	ProjTexCoord[2] = transformMatrix2 * gl_Position;
	gl_Layer = instanceID[0];
	EmitVertex();
	//-------------------------------------------------------------------
	gl_Position = gl_in[0].gl_Position + vec4( -length, length, depth, 0.0f);
	ProjTexCoord[0] = transformMatrix0 * gl_Position;
	ProjTexCoord[1] = transformMatrix1 * gl_Position;
	ProjTexCoord[2] = transformMatrix2 * gl_Position;
	gl_Layer = instanceID[0];
	EmitVertex(); 
	//-------------------------------------------------------------------
		
	gl_Position = gl_in[0].gl_Position + vec4( length, -length, depth, 0.0f);
	ProjTexCoord[0] = transformMatrix0 * gl_Position;
	ProjTexCoord[1] = transformMatrix1 * gl_Position;
	ProjTexCoord[2] = transformMatrix2 * gl_Position;
	gl_Layer = instanceID[0];
	EmitVertex();
	//--------------------------------------------------------------------

	gl_Position = gl_in[0].gl_Position + vec4( length, length, depth, 0.0f);
	ProjTexCoord[0] = transformMatrix0 * gl_Position;
	ProjTexCoord[1] = transformMatrix1 * gl_Position;
	ProjTexCoord[2] = transformMatrix2 * gl_Position;
	gl_Layer = instanceID[0];
	EmitVertex();

	EndPrimitive();
}