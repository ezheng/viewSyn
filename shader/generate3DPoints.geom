#version 420
layout(points, invocations = 2) in;
layout(points, max_vertices = 4) out;

uniform mat4 inverseMVP[2];
uniform usampler2D depthTex0;
uniform float step;

//uniform mat4 inverseMVP2;
uniform usampler2D depthTex1;



void main()
{	
	float size = 0.005;
	uint depthIndex;
	vec2 pos2D;

	if(gl_InvocationID == 0)
	{
		depthIndex = texture(depthTex0, gl_in[0].gl_Position.xy).x;
	}
	else
	{
		depthIndex = texture(depthTex1, gl_in[0].gl_Position.xy).x;
	}

	float depth = -1.0f + step * float( depthIndex );
	pos2D = vec2(gl_in[0].gl_Position.x - size, gl_in[0].gl_Position.y - size);
	gl_Position = inverseMVP[gl_InvocationID] * vec4(pos2D *2.0f - 1.0f, depth, 1.0f);
	gl_Position = gl_Position/gl_Position.w;
	EmitVertex();

	pos2D = vec2(gl_in[0].gl_Position.x - size, gl_in[0].gl_Position.y + size);
	gl_Position = inverseMVP[gl_InvocationID] * vec4(pos2D *2.0f - 1.0f, depth, 1.0f);
	gl_Position = gl_Position/gl_Position.w;
	EmitVertex();

	pos2D = vec2(gl_in[0].gl_Position.x + size, gl_in[0].gl_Position.y + size);
	gl_Position = inverseMVP[gl_InvocationID] * vec4(pos2D *2.0f - 1.0f, depth, 1.0f);
	gl_Position = gl_Position/gl_Position.w;
	EmitVertex();

	pos2D = vec2(gl_in[0].gl_Position.x + size, gl_in[0].gl_Position.y - size);
	gl_Position = inverseMVP[gl_InvocationID] * vec4(pos2D *2.0f - 1.0f, depth, 1.0f);
	gl_Position = gl_Position/gl_Position.w;
	EmitVertex();
	EndPrimitive();
}
