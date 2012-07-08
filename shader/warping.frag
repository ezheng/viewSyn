#version 420

in vec4 ProjTexCoord[3];

uniform sampler2D tex0;
uniform sampler2D tex1;
uniform sampler2D tex2;

layout(location = 0) out vec4 meanColor;

void main()
{
	vec4 projTexColor0 = vec4(0.0f, 0.0f, 0.0f, 0.0f);
	vec4 projTexColor1 = vec4(0.0f, 0.0f, 0.0f, 0.0f);
	vec4 projTexColor2 = vec4(0.0f, 0.0f, 0.0f, 0.0f);

	if( ProjTexCoord[0].z > 0.0)
		projTexColor0 = textureProj(tex0, ProjTexCoord[0]);
	if( ProjTexCoord[1].z > 0.0)
		projTexColor1 = textureProj(tex1, ProjTexCoord[1]);
	if( ProjTexCoord[2].z > 0.0)
		projTexColor2 = textureProj(tex2, ProjTexCoord[2]);
	//meanColor = (projTexColor0 + projTexColor1 + projTexColor2 );
	//meanColor.xyz = meanColor.xyz/3.0f;
	meanColor = (projTexColor0* 0.01   + projTexColor1   + projTexColor2* 0.01 );
	//if(gl_FragCoord.x<5 || gl_FragCoord.y<5)
	//	meanColor = vec4(1.0f,1.0f,0.0f,1.0f);

}


