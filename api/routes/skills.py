"""Skills 接口 — 技能列表查询"""

from fastapi import APIRouter

from core.skills import list_skills, get_skill

router = APIRouter(prefix="/skills", tags=["skills"])


@router.get("")
def get_skills():
    """返回所有可用技能"""
    return {"skills": list_skills()}


@router.get("/{name}")
def get_skill_detail(name: str):
    """返回指定技能详情"""
    skill = get_skill(name)
    if not skill:
        return {"error": f"技能 {name} 不存在"}
    return {
        "name": skill.name,
        "display_name": skill.display_name,
        "description": skill.description,
        "strategy": skill.strategy,
        "max_sub_questions": skill.max_sub_questions,
        "keywords": skill.keywords,
    }
