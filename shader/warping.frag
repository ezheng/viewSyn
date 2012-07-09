/*#version 420

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

}*/


#version 420

in vec4 ProjTexCoord[3];

uniform sampler2D tex0;
uniform sampler2D tex1;
uniform sampler2D tex2;

layout(location = 0) out float meanCost;
layout(location = 1) out vec4 meanColor;

void main()
{
	vec4 projTexColor0;
	vec4 projTexColor1;
	vec4 projTexColor2;
	bool t0 = false;
	bool t1 = false;
	bool t2 = false;
	
	vec4 baseColor = vec4(0,0,0,0);
	float numOfViews = 0;
	if(ProjTexCoord[0].x/ProjTexCoord[0].w > 0 && ProjTexCoord[0].x/ProjTexCoord[0].w <1.0 && 
			ProjTexCoord[0].y/ProjTexCoord[0].w > 0 && ProjTexCoord[0].y/ProjTexCoord[0].w < 1.0 && 
				ProjTexCoord[0].z > 0.0f )
	{
		projTexColor0 = textureProj(tex0, ProjTexCoord[0]);
		baseColor = baseColor + projTexColor0;
		numOfViews = numOfViews + 1.0;
		t0 = true;
	}
	
	if(ProjTexCoord[1].x/ProjTexCoord[1].w > 0 && ProjTexCoord[1].x/ProjTexCoord[1].w <1.0 && 
			ProjTexCoord[1].y/ProjTexCoord[1].w > 0 && ProjTexCoord[1].y/ProjTexCoord[1].w < 1.0 && 
				ProjTexCoord[1].z > 0.0 )
	{
		projTexColor1 = textureProj(tex1, ProjTexCoord[1]);
		baseColor = baseColor + projTexColor1;
		numOfViews = numOfViews + 1.0;
		t1 = true;
	}

	if(ProjTexCoord[2].x/ProjTexCoord[2].w > 0 && ProjTexCoord[2].x/ProjTexCoord[2].w <1.0 && 
		ProjTexCoord[2].y/ProjTexCoord[2].w > 0 && ProjTexCoord[2].y/ProjTexCoord[2].w < 1.0 && 
			ProjTexCoord[2].z > 0.0 )
	{
		projTexColor2 = textureProj(tex2, ProjTexCoord[2]);
		baseColor = baseColor + projTexColor2;
		numOfViews = numOfViews + 1.0;
		t2 = true;
	}
//--------------------------------------------------------------------------------------------

	meanColor = vec4(0,0,0,1.0f);
	if(numOfViews <=1)
	{
		meanCost = 1000000.0f;		
		//meanColor = vec4(1.0f,1.0f,0.0f,1.0f);
	}
	else
	{
		baseColor = baseColor/numOfViews;	

		meanCost = 0.0f;
		if(t0 == true)
		{
			meanCost = meanCost +  float(pow(projTexColor0.x - baseColor.x, 2)) + float(pow(projTexColor0.y - baseColor.y, 2)) + float(pow(projTexColor0.z - baseColor.z, 2));
			meanColor.xyz = meanColor.xyz + projTexColor0.xyz;
		}
		if(t1 == true)
		{
			meanCost = meanCost +  float(pow(projTexColor1.x - baseColor.x, 2)) + float(pow(projTexColor1.y - baseColor.y, 2)) + float(pow(projTexColor1.z - baseColor.z, 2));
			meanColor.xyz = meanColor.xyz + projTexColor1.xyz;
		}
		if(t2 == true)
		{
			meanCost = meanCost +  float(pow(projTexColor2.x - baseColor.x, 2)) + float(pow(projTexColor2.y - baseColor.y, 2)) + float(pow(projTexColor2.z - baseColor.z, 2));
			meanColor.xyz = meanColor.xyz + projTexColor2.xyz;
		}
		meanColor.xyz = meanColor.xyz/numOfViews;

	}

	//meanColor = vec4(1.0f,1.0f,0.0f,1.0f);
}





