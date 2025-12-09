"""Example 3: Trust Management

This example demonstrates trust tracking and management:
- Automatic trust tracking based on execution results
- Manual trust level setting
- Trust-based agent filtering
- Trust statistics and reporting
"""

import sys
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import random
from capabilitymesh import Mesh, TrustLevel, SimpleTrustManager


async def main():
    print("=" * 60)
    print("CapabilityMesh - Trust Management Example")
    print("=" * 60)

    # Initialize mesh with trust manager
    mesh = Mesh()

    # Register multiple agents with different reliability
    print("\n1. Registering agents...")

    def reliable_task(input_data: str) -> str:
        """Always succeeds."""
        return f"Processed: {input_data}"

    def unstable_task(input_data: str) -> str:
        """Sometimes fails."""
        if random.random() < 0.4:  # 40% failure rate
            raise ValueError("Random failure!")
        return f"Processed: {input_data}"

    def experimental_task(input_data: str) -> str:
        """Frequently fails."""
        if random.random() < 0.7:  # 70% failure rate
            raise ValueError("Experimental failure!")
        return f"Processed: {input_data}"

    await mesh.register(reliable_task, name="reliable-agent", capabilities=["task-a"])
    await mesh.register(unstable_task, name="unstable-agent", capabilities=["task-a"])
    await mesh.register(experimental_task, name="experimental-agent", capabilities=["task-a"])
    print("[OK] Registered 3 agents with different reliability")

    # Get agent IDs
    agents = await mesh.discover("task-a", limit=10)
    reliable_id = next(a.id for a in agents if a.name == "reliable-agent")
    unstable_id = next(a.id for a in agents if a.name == "unstable-agent")
    experimental_id = next(a.id for a in agents if a.name == "experimental-agent")

    # Execute multiple times to build trust history
    print("\n2. Building trust history (executing 20 times each)...")

    for agent_id, name in [
        (reliable_id, "reliable-agent"),
        (unstable_id, "unstable-agent"),
        (experimental_id, "experimental-agent"),
    ]:
        success_count = 0
        for i in range(20):
            try:
                await mesh.execute(agent_id, f"task-{i}")
                success_count += 1
            except Exception:
                pass  # Ignore failures for this demo

        print(f"  {name}: {success_count}/20 successful")

    # Check trust scores
    print("\n3. Trust scores after executions...")

    for agent_id, name in [
        (reliable_id, "reliable-agent"),
        (unstable_id, "unstable-agent"),
        (experimental_id, "experimental-agent"),
    ]:
        score = await mesh.trust.get_score(agent_id)
        print(f"\n  {name}:")
        print(f"    Level: {score.level.name}")
        print(f"    Success rate: {score.success_rate:.1%}")
        print(f"    Executions: {score.total_executions}")
        print(f"    Successes: {score.success_count}")
        print(f"    Failures: {score.failure_count}")

    # Manually set trust level
    print("\n4. Manually setting trust level...")

    # Set unstable agent to HIGH trust (override automatic calculation)
    await mesh.trust.set_level(
        unstable_id,
        TrustLevel.HIGH,
        reason="Verified by security team"
    )

    score = await mesh.trust.get_score(unstable_id)
    print(f"  [OK] Set unstable-agent to {score.level.name}")
    print(f"    Manually set: {score.manually_set}")
    print(f"    Reason: {score.metadata.get('manual_reason', 'N/A')}")

    # Trust-based filtering
    print("\n5. Discovering agents with trust filtering...")

    # Find all agents (no trust filter)
    all_agents = await mesh.discover("task-a", min_trust=None)
    print(f"\n  All agents: {len(all_agents)}")
    for agent in all_agents:
        score = await mesh.trust.get_score(agent.id)
        print(f"    - {agent.name}: {score.level.name}")

    # Find only MEDIUM+ trust agents
    medium_agents = await mesh.discover("task-a", min_trust=TrustLevel.MEDIUM)
    print(f"\n  MEDIUM+ trust agents: {len(medium_agents)}")
    for agent in medium_agents:
        score = await mesh.trust.get_score(agent.id)
        print(f"    - {agent.name}: {score.level.name}")

    # Find only HIGH+ trust agents
    high_agents = await mesh.discover("task-a", min_trust=TrustLevel.HIGH)
    print(f"\n  HIGH+ trust agents: {len(high_agents)}")
    for agent in high_agents:
        score = await mesh.trust.get_score(agent.id)
        print(f"    - {agent.name}: {score.level.name}")

    # List trusted agents
    print("\n6. Listing trusted agents...")

    trusted = await mesh.trust.list_trusted_agents(min_level=TrustLevel.MEDIUM)
    print(f"\n  Trusted agents (MEDIUM+): {len(trusted)}")
    for score in trusted:
        agent_name = next(a.name for a in agents if a.id == score.agent_id)
        print(f"    - {agent_name}")
        print(f"      Level: {score.level.name}")
        print(f"      Success rate: {score.success_rate:.1%}")

    # Trust statistics
    print("\n7. Overall trust statistics...")

    stats = await mesh.trust.get_stats()
    print(f"\n  Total agents tracked: {stats['total_agents']}")
    print(f"  Total executions: {stats['total_executions']}")
    print(f"  Overall success rate: {stats['overall_success_rate']:.1%}")
    print("\n  Trust level distribution:")
    for level_name, count in stats['level_distribution'].items():
        if count > 0:
            print(f"    {level_name}: {count} agents")

    # Reset trust score
    print("\n8. Resetting trust score...")

    await mesh.trust.reset_agent(experimental_id)
    score = await mesh.trust.get_score(experimental_id)
    print(f"  [OK] Reset experimental-agent")
    print(f"    Level: {score.level.name}")
    print(f"    Executions: {score.total_executions}")

    # Demonstrate trust progression
    print("\n9. Demonstrating trust level progression...")

    def new_task(input_data: str) -> str:
        return f"Result: {input_data}"

    await mesh.register(new_task, name="new-agent", capabilities=["task-b"])

    new_agents = await mesh.discover("task-b", limit=1)
    new_id = new_agents[0].id

    print("\n  Initial state:")
    score = await mesh.trust.get_score(new_id)
    print(f"    Level: {score.level.name}, Executions: {score.total_executions}")

    print("\n  After 3 successful executions:")
    for i in range(3):
        await mesh.execute(new_id, f"data-{i}")
    score = await mesh.trust.get_score(new_id)
    print(f"    Level: {score.level.name}, Executions: {score.total_executions}")

    print("\n  After 8 successful executions:")
    for i in range(5):
        await mesh.execute(new_id, f"data-{i}")
    score = await mesh.trust.get_score(new_id)
    print(f"    Level: {score.level.name}, Executions: {score.total_executions}")

    print("\n  After 15 successful executions:")
    for i in range(7):
        await mesh.execute(new_id, f"data-{i}")
    score = await mesh.trust.get_score(new_id)
    print(f"    Level: {score.level.name}, Executions: {score.total_executions}")

    print("\n  After 20 successful executions:")
    for i in range(5):
        await mesh.execute(new_id, f"data-{i}")
    score = await mesh.trust.get_score(new_id)
    print(f"    Level: {score.level.name}, Executions: {score.total_executions}")
    print(f"    Success rate: {score.success_rate:.1%}")

    # Trust levels explanation
    print("\n" + "=" * 60)
    print("Trust Level Criteria")
    print("=" * 60)
    print("\nUNTRUSTED (0):")
    print("  - Never executed or explicitly untrusted")
    print("\nLOW (1):")
    print("  - < 50% success rate OR < 5 executions")
    print("\nMEDIUM (2):")
    print("  - 50-80% success rate, >= 5 executions")
    print("\nHIGH (3):")
    print("  - 80-95% success rate, >= 10 executions")
    print("\nVERIFIED (4):")
    print("  - > 95% success rate, >= 20 executions")
    print("  - OR manually verified by administrator")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
